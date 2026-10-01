"""Reteste de correcao de redacao via APIs hospedadas gratuitas.

Um cliente unico para Groq, Cerebras e Gemini (todos tem endpoint compativel com
a API da OpenAI), com retry/backoff em 429, respeito a rate limit e checkpoint
por redacao (retoma de onde parou).

Dois modos de prompt:
  holistico : uma chamada, JSON com C1..C5 + total          (comparavel ao v5)
  mts       : uma chamada por competencia, com a rubrica do  (mais discriminacao)
              nivel no prompt; a justificativa serve de insumo de feedback

Saida (um CSV por modelo/modo), colunas:
  index_redacao, score, pred_c1..pred_c5, pred_total, ok, modelo, modo, ts, raw

Chaves de API por variavel de ambiente:
  GROQ_API_KEY, CEREBRAS_API_KEY, GEMINI_API_KEY

Uso:
  set GEMINI_API_KEY=...        (Windows)   /   export GEMINI_API_KEY=...  (bash)
  python run_api_scoring.py --provider gemini --model gemini-2.5-flash \
      --modo holistico --amostra data/amostra_300.csv --out results/api/gemini_holistico.csv
  python run_api_scoring.py --provider groq --model openai/gpt-oss-120b --modo mts --limit 20
  python run_api_scoring.py            # auto-teste do parser, sem rede
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

PROVIDERS = {
    "groq":     {"url": "https://api.groq.com/openai/v1/chat/completions",     "env": "GROQ_API_KEY",     "rpm": 30, "reasoning": "low"},
    "cerebras": {"url": "https://api.cerebras.ai/v1/chat/completions",          "env": "CEREBRAS_API_KEY", "rpm": 5,  "reasoning": "low"},
    "gemini":   {"url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                 "env": "GEMINI_API_KEY", "rpm": 10, "reasoning": "none"},
    # vLLM local (Colab). Sem limite real; a chave e qualquer string (vllm serve sem --api-key).
    "vllm":     {"url": "http://localhost:8000/v1/chat/completions",           "env": "VLLM_API_KEY",     "rpm": 6000, "reasoning": "low"},
}

COMPS = ["C1", "C2", "C3", "C4", "C5"]

RUBRICA = {
    "C1": "Dominio da modalidade escrita formal da lingua portuguesa (ortografia, "
          "concordancia, regencia, pontuacao, escolha de registro).",
    "C2": "Compreensao da proposta e aplicacao do tipo textual dissertativo-argumentativo "
          "para desenvolver o tema, sem tangenciar nem fugir.",
    "C3": "Selecao, relacao, organizacao e interpretacao de informacoes, fatos, opinioes e "
          "argumentos em defesa de um ponto de vista (projeto de texto e autoria).",
    "C4": "Conhecimento dos mecanismos linguisticos de coesao: articulacao entre paragrafos, "
          "periodos e partes do texto (conectivos, referenciacao).",
    "C5": "Elaboracao de proposta de intervencao para o problema, detalhada e respeitando os "
          "direitos humanos, com agente, acao, modo/meio, efeito e detalhamento.",
}
NIVEIS = ("0 = ausente/desconsiderando o solicitado; 40 = precario; 80 = mediano com falhas; "
          "120 = mediano; 160 = bom com poucas falhas; 200 = excelente, sem falhas relevantes")

_last_call: dict[str, float] = {}
_drop: dict[str, set] = {}  # "provider/model" -> campos opcionais que o modelo rejeita
KEY_ENV: str | None = None  # sobrescreve o nome da variavel da chave (--key-env)
_anchors: dict[str, str] | None = None  # comp -> bloco de exemplos, carregado sob demanda
RUBRICA_C5_PATH: str | None = None  # --rubrica-c5, texto da rubrica de C5 do modo mts_rr
LT_PATH = "data/lt_amostra_300.csv"  # --lt-features, contagem do LanguageTool do modo mts_lt
_lt: pd.DataFrame | None = None
ANCHORS_PATH = "data/anchors.csv"  # --anchors, ancoras dos modos few-shot (build_anchors.py)
MAX_TOKENS: int | None = None  # --max-tokens, sobrescreve o limite por chamada dos modos MTS

# Instrucao de C5 do mts_fs2. E a versao 0 do Reflect-and-Revise (reflect_revise.py).
C5_RUBRICA_V0 = (
    "Passo 1: verifique cada elemento na proposta e marque presente ou ausente: "
    "agente (quem faz), acao (o que faz), modo/meio (como), efeito (para que), "
    "detalhamento (explica algum elemento).\n"
    "Passo 2: conte os elementos validos e articulados ao problema. A nota segue a "
    "contagem, NAO e tudo ou nada: 200 os 5; 160 quatro; 120 tres; 80 dois; 40 um; "
    "0 so quando nao ha proposta alguma ou ela fere direitos humanos."
)


def _bloco_anchors(comp):
    global _anchors
    if _anchors is None:
        a = pd.read_csv(ANCHORS_PATH)
        _anchors = {c: "\n".join(f"EXEMPLO ({c} = {r.band}):\n{r.essay}\n"
                                 for r in a[a.comp == c].itertuples()) for c in COMPS}
    return _anchors[comp]


def _get_key(provider):
    env_name = KEY_ENV or PROVIDERS[provider]["env"]
    key = os.environ.get(env_name)
    if not key:
        sys.exit(f"variavel de ambiente {env_name} nao definida")
    return key


def probe(provider, model):
    """Manda 1 requisicao trivial e imprime status + corpo cru. Diagnostico."""
    cfg = PROVIDERS[provider]
    key = _get_key(provider)
    body = {"model": model,
            "messages": [{"role": "user", "content": 'Responda so com JSON: {"ok": 1}'}],
            "max_tokens": 1500, "temperature": 0}
    r = requests.post(cfg["url"], json=body, headers={"Authorization": f"Bearer {key}"}, timeout=60)
    print(f"HTTP {r.status_code}\n{r.text}")


def listar_modelos(provider):
    cfg = PROVIDERS[provider]
    key = _get_key(provider)
    if provider == "gemini":
        url = "https://generativelanguage.googleapis.com/v1beta/models?key=" + key
        r = requests.get(url, timeout=60).json()
        nomes = [m["name"].replace("models/", "") for m in r.get("models", [])]
    else:
        base = cfg["url"].replace("/chat/completions", "/models")
        r = requests.get(base, headers={"Authorization": f"Bearer {key}"}, timeout=60).json()
        nomes = [m["id"] for m in r.get("data", [])]
    print("\n".join(sorted(nomes)) or r)


def limpar(texto):
    if pd.isna(texto):
        return ""
    texto = str(texto).strip("[]'\" ").replace("\\n", " ").replace("\n", " ")
    texto = re.sub(r"\[[A-Z/]+\]", "", texto)
    texto = re.sub(r"\{[a-z]+\}", "", texto)
    return re.sub(r"\s+", " ", texto).strip()


def chat(provider, model, prompt, temperature=0.1, max_tokens=400, max_retries=6):
    cfg = PROVIDERS[provider]
    key = _get_key(provider)

    wait = 60.0 / cfg["rpm"]
    since = time.time() - _last_call.get(provider, 0.0)
    if since < wait:
        time.sleep(wait - since)

    ban = _drop.setdefault(f"{provider}/{model}", set())
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if "response_format" not in ban:
        body["response_format"] = {"type": "json_object"}
    if cfg.get("reasoning") and "reasoning_effort" not in ban:
        body["reasoning_effort"] = cfg["reasoning"]  # desliga/reduz "thinking"
    headers = {"Authorization": f"Bearer {key}"}
    for attempt in range(max_retries):
        _last_call[provider] = time.time()
        try:
            r = requests.post(cfg["url"], json=body, headers=headers, timeout=120)
            if r.status_code == 200:
                j = r.json()
                ch = (j.get("choices") or [{}])[0]
                content = (ch.get("message") or {}).get("content") or ""
                if not content.strip():
                    print(f"  200 sem conteudo. finish_reason={ch.get('finish_reason')} "
                          f"body={json.dumps(j)[:700]}", file=sys.stderr)
                return content
            if r.status_code == 400:
                for campo in ("reasoning_effort", "response_format"):
                    if campo in body:
                        print(f"  400, desativando {campo} para {model}. body={r.text[:200]}", file=sys.stderr)
                        body.pop(campo)
                        ban.add(campo)
                        break
                else:
                    sys.exit(f"{provider}/{model}: 400 sem recuperacao - {r.text[:400]}")
                continue
            if r.status_code in (401, 402, 403, 404):
                sys.exit(f"{provider}/{model}: HTTP {r.status_code} - {r.text[:200]}")
            if r.status_code in (429, 500, 502, 503):
                ra = r.headers.get("retry-after")
                sleep = float(ra) if ra else (2 ** attempt + attempt)
                print(f"  {r.status_code}, aguardando {sleep:.0f}s", file=sys.stderr)
                time.sleep(sleep)
                continue
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  erro de rede: {e}, tentativa {attempt + 1}", file=sys.stderr)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"{provider}/{model}: falhou apos {max_retries} tentativas")


def prompt_holistico(essay):
    return (
        "Voce e avaliador oficial de redacoes do ENEM. Avalie a redacao nas 5 "
        "competencias, cada uma em 0, 40, 80, 120, 160 ou 200 pontos.\n\n"
        + "\n".join(f"- {c}: {RUBRICA[c]}" for c in COMPS)
        + f"\n\nEscala por competencia: {NIVEIS}\n\nREDACAO:\n{essay}\n\n"
        'Responda APENAS com JSON: {"C1":v,"C2":v,"C3":v,"C4":v,"C5":v,"Nota_Total":soma}'
    )


def prompt_mts(essay, comp):
    return (
        f"Voce e avaliador oficial de redacoes do ENEM. Avalie SOMENTE a competencia {comp}.\n\n"
        f"{comp}: {RUBRICA[comp]}\n"
        f"Pontue em 0, 40, 80, 120, 160 ou 200. Referencia: {NIVEIS}.\n\n"
        f"REDACAO:\n{essay}\n\n"
        f'Responda APENAS com JSON: {{"{comp}": valor, "justificativa": "1 a 3 frases"}}'
    )


def prompt_mts_fs(essay, comp, extra=""):
    """MTS com exemplos ancora (few-shot) por faixa de nota, um por faixa."""
    return (
        f"Voce e avaliador oficial de redacoes do ENEM. Avalie SOMENTE a competencia {comp}.\n\n"
        f"{comp}: {RUBRICA[comp]}\n"
        f"Pontue em 0, 40, 80, 120, 160 ou 200.\n\n"
        f"Use os exemplos abaixo para calibrar a escala. Note que notas parciais existem: "
        f"uma resposta incompleta nao e automaticamente 0.\n\n"
        f"{_bloco_anchors(comp)}\n"
        f"{extra}"
        f"REDACAO A AVALIAR:\n{essay}\n\n"
        f'Responda APENAS com JSON: {{"{comp}": valor, "justificativa": "1 a 3 frases"}}'
    )


def prompt_mts_fs2(essay, comp):
    """mts_fs para C1-C4; para C5, ancoras + checklist dos 5 elementos (opcao A)."""
    if comp != "C5":
        return prompt_mts_fs(essay, comp)
    return prompt_c5(essay, C5_RUBRICA_V0)


def prompt_mts_rr(essay, comp):
    """mts_fs2 com a rubrica de C5 lida de arquivo (--rubrica-c5), refinada por Reflect-and-Revise."""
    if comp != "C5":
        return prompt_mts_fs(essay, comp)
    with open(RUBRICA_C5_PATH, encoding="utf-8") as fh:
        return prompt_c5(essay, fh.read().strip())


def bloco_lt(idx):
    """Contagem de desvios do LanguageTool (lt_features.py) como contexto para C1."""
    global _lt
    if _lt is None:
        _lt = pd.read_csv(LT_PATH).set_index("index_redacao")
    f = _lt.loc[idx]
    mediana = _lt["desvios_100"].median()
    ex = "; ".join(f"'{e['trecho']}' ({e['msg']})" for e in json.loads(f["exemplos"])[:6])
    return (
        f"APOIO: um corretor automatico (LanguageTool) encontrou {int(f['desvios'])} possiveis "
        f"desvios em {int(f['palavras'])} palavras, {f['desvios_100']:.1f} por 100 palavras "
        f"(mediana das redacoes: {mediana:.1f}). Ortografia {int(f['ortografia'])}, gramatica "
        f"{int(f['gramatica'])}, pontuacao {int(f['pontuacao'])}, maiusculas "
        f"{int(f['maiusculas'])}, palavras confundidas {int(f['palavras_confundidas'])}. "
        f"Exemplos: {ex or 'nenhum'}. O corretor tem falsos positivos e nao ve todos os "
        f"desvios: use como apoio e confira no texto.\n\n"
    )


def prompt_mts_lt(essay, comp, idx):
    """mts_fs2 com a contagem do LanguageTool no prompt de C1."""
    if comp != "C1":
        return prompt_mts_fs2(essay, comp)
    return prompt_mts_fs(essay, comp, bloco_lt(idx))


def prompt_c5(essay, rubrica):
    return (
        "Voce e avaliador oficial de redacoes do ENEM. Avalie SOMENTE a competencia C5 "
        "(proposta de intervencao para o problema, respeitando os direitos humanos).\n\n"
        f"{rubrica}\n\n"
        "Exemplos para calibrar a escala:\n\n"
        f"{_bloco_anchors('C5')}\n"
        f"REDACAO A AVALIAR:\n{essay}\n\n"
        'Responda APENAS com JSON: {"C5": valor, "elementos": {"agente": true, "acao": '
        'true, "meio": true, "efeito": true, "detalhamento": true}, '
        '"justificativa": "1 a 3 frases"}'
    )


def prompt_mts_v2(essay, comp):
    """Igual ao mts para C2/C3/C4; para C1 e C5 pede analise estruturada antes da nota."""
    if comp == "C1":
        return (
            "Voce e avaliador oficial de redacoes do ENEM. Avalie SOMENTE a competencia C1 "
            "(dominio da norma culta escrita).\n\n"
            "Passo 1: liste ate 8 desvios representativos, por tipo: ortografia, acentuacao, "
            "pontuacao, concordancia, regencia, crase, colocacao, registro.\n"
            "Passo 2: pontue C1 pela densidade e gravidade dos desvios em relacao ao tamanho "
            "do texto: 200 sem desvios ou raros; 160 poucos desvios; 120 alguns desvios sem "
            "comprometer; 80 muitos desvios; 40 desvios sistematicos; 0 dominio precario.\n\n"
            f"REDACAO:\n{essay}\n\n"
            'Responda APENAS com JSON: {"C1": valor, "desvios": ["..."], '
            '"justificativa": "1 a 3 frases"}'
        )
    if comp == "C5":
        return (
            "Voce e avaliador oficial de redacoes do ENEM. Avalie SOMENTE a competencia C5 "
            "(proposta de intervencao para o problema, respeitando os direitos humanos).\n\n"
            "Passo 1: verifique cada um dos 5 elementos na proposta: agente (quem faz), acao "
            "(o que faz), modo/meio (como), efeito (para que), detalhamento (explica algum "
            "elemento). Marque presente ou ausente.\n"
            "Passo 2: pontue C5 pelo numero de elementos validos e bem articulados ao "
            "problema: 200 os 5; 160 quatro; 120 tres; 80 dois; 40 um; 0 nenhum ou proposta "
            "ausente ou que fere direitos humanos.\n\n"
            f"REDACAO:\n{essay}\n\n"
            'Responda APENAS com JSON: {"C5": valor, "elementos": {"agente": true, "acao": '
            'true, "meio": true, "efeito": true, "detalhamento": true}, '
            '"justificativa": "1 a 3 frases"}'
        )
    return prompt_mts(essay, comp)


def _num(v):
    """Valor cru -> faixa do ENEM (0..200, passo 40). Corrige escala 0-20 -> 0-200."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if 0 < f <= 20:  # modelo usou escala 0-20 (heuristica do v5)
        f *= 10
    return min(max(int(round(f / 40.0)) * 40, 0), 200)


def parse_notas(text, comp=None):
    """Extrai dict com C1..C5 (holistico) ou {comp: nota, justificativa} (mts)."""
    try:
        t = re.sub(r"<thought>.*?</thought>|<think(ing)?>.*?</think(ing)?>|```json|```",
                   "", str(text), flags=re.DOTALL)
        i, j = t.find("{"), t.rfind("}") + 1
        d = json.loads(t[i:j]) if 0 <= i < j else {}
    except (ValueError, json.JSONDecodeError):
        d = {}
    if comp:
        n = _num(d.get(comp))
        return {comp: n, "justificativa": str(d.get("justificativa", ""))} if n is not None else None
    vals = {c: _num(d.get(c)) for c in COMPS}
    if any(v is None for v in vals.values()):
        return None
    vals["Nota_Total"] = sum(vals[c] for c in COMPS)
    return vals


def score_redacao(provider, model, essay, modo, temperature, comps=COMPS, idx=None):
    if modo == "holistico":
        resp = chat(provider, model, prompt_holistico(essay), temperature, 2048)
        notas = parse_notas(resp)
        raw = "" if notas else str(resp)  # guarda a resposta crua so quando o parse falha
    else:
        gerar = {"mts2": prompt_mts_v2, "mts_fs": prompt_mts_fs,
                 "mts_fs2": prompt_mts_fs2, "mts_rr": prompt_mts_rr,
                 "mts_lt": lambda e, c: prompt_mts_lt(e, c, idx)}.get(modo, prompt_mts)
        maxtok = MAX_TOKENS or (800 if modo in ("mts2", "mts_fs2", "mts_rr", "mts_lt") else 512)
        notas, partes = {}, []
        for c in comps:
            resp = chat(provider, model, gerar(essay, c), temperature, maxtok)
            p = parse_notas(resp, comp=c)
            if p is None:
                return None, "|".join(partes)
            notas[c] = p[c]
            partes.append(f"{c}: {p['justificativa']}")
        notas["Nota_Total"] = sum(notas[c] for c in COMPS) if len(notas) == len(COMPS) else ""
        raw = " || ".join(partes)
    if notas is None:
        return None, ""
    return notas, raw


def run(amostra, provider, model, modo, out, temperature, limit, offset=0, comps=COMPS):
    df = pd.read_csv(amostra)
    df = df.iloc[offset: offset + limit if limit else None]
    print(f"fatia: linhas {offset} a {offset + len(df) - 1} ({len(df)} redacoes)")
    feitos = set()
    if os.path.exists(out):
        prev = pd.read_csv(out)
        outros = prev[(prev["modelo"] != model) | (prev["modo"] != modo)]
        if len(outros):  # retomar por cima de outro modelo/modo mistura resultados
            sys.exit(f"{out} tem {len(outros)} linhas de outro modelo/modo "
                     f"({outros['modelo'].iloc[0]}, {outros['modo'].iloc[0]}). Use outro --out.")
        feitos = set(prev["index_redacao"])
        print(f"retomando: {len(feitos)} ja feitas")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    novo = not os.path.exists(out)
    with open(out, "a", encoding="utf-8", newline="") as fh:
        if novo:
            fh.write("index_redacao,score,c1,c2,c3,c4,c5,pred_c1,pred_c2,pred_c3,pred_c4,pred_c5,"
                     "pred_total,ok,modelo,modo,ts,raw\n")
        for _, row in df.iterrows():
            idx = int(row["index_redacao"])
            if idx in feitos:
                continue
            try:
                notas, raw = score_redacao(provider, model, limpar(row["essay"]), modo, temperature, comps, idx)
            except RuntimeError as e:
                print(f"[{idx}] {e}, pulando pra proxima", file=sys.stderr)
                continue  # esgotou retry nessa redacao, nao aborta a fatia toda
            ok = notas is not None
            pc = [notas.get(c, "") if ok else "" for c in COMPS]
            pt = notas["Nota_Total"] if ok else ""
            ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            raw_c = '"' + raw.replace('"', "'")[:500] + '"'
            g = [row["c1"], row["c2"], row["c3"], row["c4"], row["c5"]]
            fh.write(f"{idx},{row['score']},{g[0]},{g[1]},{g[2]},{g[3]},{g[4]},"
                     f"{pc[0]},{pc[1]},{pc[2]},{pc[3]},{pc[4]},"
                     f"{pt},{int(ok)},{model},{modo},{ts},{raw_c}\n")
            fh.flush()
            print(f"[{idx}] {'ok ' + str(pt) if ok else 'FALHA parse'}")
    print(f"\nfeito. saida em {out}  (rode: python evaluate.py {out})")


def demo():
    assert parse_notas('{"C1":160,"C2":120,"C3":120,"C4":160,"C5":80,"Nota_Total":640}')["Nota_Total"] == 640
    assert parse_notas('lixo {"C1":16,"C2":12,"C3":12,"C4":16,"C5":8} fim')["C1"] == 160  # escala 0-20
    assert parse_notas("```json\n{\"C1\":200,\"C2\":200,\"C3\":200,\"C4\":200,\"C5\":200}\n```")["Nota_Total"] == 1000
    assert parse_notas('{"C1":200}') is None  # incompleto
    assert parse_notas('{"C3": 130, "justificativa": "ok"}', comp="C3")["C3"] == 120  # arredonda p/ faixa
    assert parse_notas("sem json", comp="C1") is None
    print("demo ok: parser aceita JSON sujo, corrige escala 0-20, arredonda para faixa de 40, "
          "rejeita incompleto.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=PROVIDERS)
    ap.add_argument("--model")
    ap.add_argument("--modo", choices=["holistico", "mts", "mts2", "mts_fs", "mts_fs2", "mts_rr", "mts_lt"],
                    default="holistico")
    ap.add_argument("--amostra", default="data/amostra_300.csv")
    ap.add_argument("--out")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0, help="processa no maximo N redacoes")
    ap.add_argument("--offset", type=int, default=0, help="pula as primeiras N (dividir trabalho entre contas)")
    ap.add_argument("--rpm", type=int, help="sobrescreve o RPM padrao do provedor")
    ap.add_argument("--reasoning", help="sobrescreve o reasoning_effort do provedor (low, medium, high)")
    ap.add_argument("--anchors", default="data/anchors.csv",
                    help="ancoras dos modos few-shot (ex data/anchors_k3.csv, 3 por faixa)")
    ap.add_argument("--max-tokens", type=int, help="limite de tokens por chamada (raciocinio conta)")
    ap.add_argument("--list-models", action="store_true", help="lista os modelos visiveis pela chave")
    ap.add_argument("--probe", action="store_true", help="1 requisicao de teste, imprime resposta crua")
    ap.add_argument("--key-env", help="nome da variavel com a chave (ex GEMINI_API_KEY_2), varias contas")
    ap.add_argument("--comps", default=",".join(COMPS),
                    help="competencias a pontuar nos modos MTS (ex C5); as outras ficam vazias")
    ap.add_argument("--rubrica-c5", help="arquivo com a rubrica de C5 (modo mts_rr)")
    ap.add_argument("--lt-features", default="data/lt_amostra_300.csv",
                    help="saida do lt_features.py (modo mts_lt)")
    args = ap.parse_args(argv)
    global KEY_ENV, RUBRICA_C5_PATH, LT_PATH, MAX_TOKENS, ANCHORS_PATH
    KEY_ENV = args.key_env
    RUBRICA_C5_PATH = args.rubrica_c5
    LT_PATH = args.lt_features
    MAX_TOKENS = args.max_tokens
    ANCHORS_PATH = args.anchors
    comps = [c.strip().upper() for c in args.comps.split(",")]
    if args.modo == "mts_rr" and not args.rubrica_c5:
        sys.exit("modo mts_rr precisa de --rubrica-c5")

    if not args.provider:
        demo()
        return
    if args.list_models:
        listar_modelos(args.provider)
        return
    if args.probe:
        probe(args.provider, args.model or "gemini-2.5-flash-lite")
        return
    if not args.model:
        sys.exit("informe --model")
    if args.rpm:
        PROVIDERS[args.provider]["rpm"] = args.rpm
    if args.reasoning:
        PROVIDERS[args.provider]["reasoning"] = args.reasoning
    out = args.out or f"results/api/{args.provider}_{args.modo}.csv"
    run(args.amostra, args.provider, args.model, args.modo, out, args.temperature, args.limit,
        args.offset, comps)


if __name__ == "__main__":
    main()
