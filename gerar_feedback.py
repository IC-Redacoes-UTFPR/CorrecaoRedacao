"""Feedback formativo em duas etapas: a nota ja vem pronta e uma chamada por redacao escreve o
feedback das 5 competencias juntas.

Separar nota e feedback permite medir a qualidade do feedback sem o erro de nota (--notas
humano) e tambem o sistema real (--notas com o CSV de predicao do run_api_scoring.py). Ver
todas as competencias de uma vez evita citar o mesmo trecho em todas.

Cada problema citado e conferido contra o texto da redacao: "literal" fica false quando o
trecho nao aparece no texto (o modelo inventou ou corrigiu o erro do aluno ao copiar).

Uso:
    python gerar_feedback.py --provider vllm --model openai/gpt-oss-120b --notas humano \\
        --out results/colab/feedback_humano.jsonl --limit 5
    python gerar_feedback.py --provider vllm --model openai/gpt-oss-120b \\
        --notas results/colab/gptoss120b_mts_fs.csv --out results/colab/feedback_pred.jsonl
    python gerar_feedback.py --relatorio results/colab/feedback_humano.jsonl   # gera o .md
    python gerar_feedback.py                                                  # auto-teste
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import pandas as pd

import run_api_scoring as ras
from run_api_scoring import COMPS, RUBRICA, chat, extrair_json, limpar


def norm(s):
    """Minusculas, sem pontuacao e espaco unico: tolera diferenca de virgula e espaco ao citar."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(s).lower())).strip()


def prompt_feedback(essay, notas, tema_id=None):
    comps = "\n".join(f"- {c} ({notas[c]} de 200): {RUBRICA[c]}" for c in COMPS)
    # so o titulo do tema: com o texto motivador no prompt o modelo citava trechos dele como se
    # fossem do aluno (6 dos 8 trechos nao literais do smoke de 2026-10-01)
    tema = f"TEMA DA REDACAO: {ras.TEMAS[int(tema_id)][0]}\n\n" if ras.TEMAS and tema_id is not None else ""
    item = ('{"pontos_fortes": ["..."], "problemas": [{"trecho": "...", "explicacao": "...", '
            '"correcao": "..."}], "como_melhorar": ["..."]}')
    return (
        "Voce e professor de redacao do ENEM e vai escrever um feedback formativo para o "
        "estudante. A redacao ja foi corrigida; as notas abaixo sao finais, nao as reavalie. "
        "Explique ao estudante por que ele recebeu cada nota e como melhorar.\n\n"
        f"{tema}"
        f"NOTAS POR COMPETENCIA:\n{comps}\n\n"
        f"REDACAO:\n{essay}\n\n"
        "Para cada competencia, falando diretamente com o estudante (voce), em portugues:\n"
        "- pontos_fortes: 1 ou 2 coisas que a redacao faz bem nesta competencia.\n"
        "- problemas: ate 3, os que mais pesaram na nota. Em \"trecho\", copie LITERALMENTE um "
        "trecho curto da redacao, com os erros do estudante exatamente como estao, sem corrigir "
        "nada. Em \"explicacao\", diga o que esta errado e por que, nos termos da competencia. "
        "Em \"correcao\", reescreva o trecho do jeito certo (obrigatorio na C1; nas outras, uma "
        "versao melhorada ou \"\").\n"
        "- como_melhorar: 1 a 3 acoes concretas para a proxima redacao, ligadas aos problemas.\n"
        "Use trechos diferentes em cada competencia, cada um mostrando um problema daquela "
        "competencia. O tom e a gravidade devem acompanhar a nota: nota alta, problemas "
        "pequenos; com nota 200, problemas pode ser lista vazia.\n\n"
        "Responda APENAS com JSON: {" + ", ".join(f'"{c}": {item}' for c in COMPS) + "}"
    )


def marcar_literal(fb, essay):
    """Poe literal true/false em cada problema. Devolve (literais, total)."""
    texto, lit, tot = norm(essay), 0, 0
    for c in COMPS:
        for p in (fb.get(c) or {}).get("problemas") or []:
            p["literal"] = bool(norm(p.get("trecho", ""))) and norm(p.get("trecho", "")) in texto
            lit, tot = lit + p["literal"], tot + 1
    return lit, tot


def notas_de(fonte, df):
    """index_redacao -> {C1..C5}. fonte = 'humano' (c1..c5 da amostra) ou CSV de predicao."""
    if fonte == "humano":
        return {int(r.index_redacao): {c: int(getattr(r, c.lower())) for c in COMPS} for r in df.itertuples()}
    p = pd.read_csv(fonte).set_index("index_redacao")
    return {int(i): {c: int(p.loc[i, f"pred_{c.lower()}"]) for c in COMPS}
            for i in df["index_redacao"] if i in p.index and p.loc[i, [f"pred_{c.lower()}" for c in COMPS]].notna().all()}


def run(amostra, fonte, provider, model, out, limit=0, max_tokens=6000):
    df = pd.read_csv(amostra)
    df = df.head(limit) if limit else df
    notas = notas_de(fonte, df)
    feitos = set()
    if os.path.exists(out):
        prev = [json.loads(l) for l in open(out, encoding="utf-8")]
        outros = [r for r in prev if (r["modelo"], r["notas_fonte"]) != (model, fonte)]
        if outros:  # mesmo cuidado do run_api_scoring: nao misturar rodadas diferentes
            sys.exit(f"{out} tem linhas de outro modelo/fonte de nota ({outros[0]['modelo']}, "
                     f"{outros[0]['notas_fonte']}). Use outro --out.")
        feitos = {r["index_redacao"] for r in prev}
        print(f"retomando: {len(feitos)} ja feitas")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    lit_tot = [0, 0]
    for row in df.itertuples():
        idx = int(row.index_redacao)
        if idx in feitos or idx not in notas:
            continue
        essay = limpar(row.essay)
        try:
            resp = chat(provider, model, prompt_feedback(essay, notas[idx], row.prompt), 0.1, max_tokens)
        except RuntimeError as e:
            print(f"[{idx}] {e}, pulando", file=sys.stderr)
            continue
        fb = extrair_json(resp)
        if not all(isinstance(fb.get(c), dict) for c in COMPS):
            print(f"[{idx}] FALHA parse")
            continue
        lit, tot = marcar_literal(fb, essay)
        lit_tot = [lit_tot[0] + lit, lit_tot[1] + tot]
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"index_redacao": idx, "modelo": model, "notas_fonte": fonte,
                                 "notas": notas[idx], "feedback": fb}, ensure_ascii=False) + "\n")
        print(f"[{idx}] ok, trechos literais {lit}/{tot}")
    if lit_tot[1]:
        print(f"\ntrechos literais nesta rodada: {lit_tot[0]}/{lit_tot[1]} "
              f"({100 * lit_tot[0] / lit_tot[1]:.0f}%)")
    print(f"feito. saida em {out}  (relatorio: python gerar_feedback.py --relatorio {out})")


def relatorio(jsonl, amostra, temas_path):
    """Markdown legivel: tema, texto, notas humano x usadas, feedback e marca de trecho nao literal."""
    am = pd.read_csv(amostra).set_index("index_redacao")
    temas = pd.read_csv(temas_path).set_index("id")["title"]
    linhas = [f"# Feedback: {os.path.basename(jsonl)}\n",
              "Trechos marcados com [NAO LITERAL] nao aparecem exatamente assim na redacao.\n"]
    lit = tot = 0
    for r in (json.loads(l) for l in open(jsonl, encoding="utf-8")):
        i, a = r["index_redacao"], am.loc[r["index_redacao"]]
        linhas += [f"\n---\n\n## Redacao {i}\n", f"**Tema:** {temas.get(a['prompt'], '?')}\n",
                   "| | C1 | C2 | C3 | C4 | C5 | Total |", "|---|---|---|---|---|---|---|",
                   "| Humano | " + " | ".join(str(a[c.lower()]) for c in COMPS) + f" | {a['score']} |",
                   f"| Usada no feedback ({r['notas_fonte'] if r['notas_fonte'] == 'humano' else 'modelo'}) | "
                   + " | ".join(str(r["notas"][c]) for c in COMPS) + f" | {sum(r['notas'].values())} |\n",
                   f"**Texto:**\n\n> {limpar(a['essay'])}\n"]
        for c in COMPS:
            f = r["feedback"].get(c) or {}
            linhas.append(f"\n### {c} (nota {r['notas'][c]})\n")
            linhas.append("**Pontos fortes:**\n" + "\n".join(f"- {p}" for p in f.get("pontos_fortes") or []) + "\n")
            linhas.append("**Problemas:**")
            for p in f.get("problemas") or []:
                tot += 1
                lit += p.get("literal", False)
                tag = "" if p.get("literal") else " [NAO LITERAL]"
                cor = f"\n  Correcao: \"{p['correcao']}\"" if p.get("correcao") else ""
                linhas.append(f"- \"{p.get('trecho')}\"{tag}\n  {p.get('explicacao')}{cor}")
            linhas.append("\n**Como melhorar:**\n" + "\n".join(f"- {p}" for p in f.get("como_melhorar") or []) + "\n")
    linhas.insert(2, f"Trechos literais: {lit}/{tot}.\n")
    md = os.path.splitext(jsonl)[0] + ".md"
    open(md, "w", encoding="utf-8").write("\n".join(linhas))
    print(f"relatorio em {md} (trechos literais {lit}/{tot})")


def demo():
    essay = "O governo deve agir, pois ninguem aceitavam isso na sociedade."
    fb = {c: {"pontos_fortes": [], "problemas": [], "como_melhorar": []} for c in COMPS}
    fb["C1"]["problemas"] = [{"trecho": "ninguem aceitavam isso", "explicacao": "concordancia"},
                             {"trecho": "ninguem aceitava isso", "explicacao": "copia corrigida"}]
    assert marcar_literal(fb, essay) == (1, 2)
    assert fb["C1"]["problemas"][1]["literal"] is False  # corrigir ao copiar conta como nao literal
    assert extrair_json('```json\n{"C1": {"problemas": []}}\n```')["C1"] == {"problemas": []}
    print("demo ok: checagem de literalidade pega trecho corrigido ao copiar.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=ras.PROVIDERS)
    ap.add_argument("--model")
    ap.add_argument("--key-env", help="nome da variavel com a chave (ex GEMINI_API_KEY_2)")
    ap.add_argument("--amostra", default="data/amostra_feedback_30.csv")
    ap.add_argument("--notas", default="humano", help="'humano' ou CSV de predicao do run_api_scoring.py")
    ap.add_argument("--temas", default="data/prompts_essaybr.csv")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--relatorio", help="JSONL ja gerado: escreve o .md legivel ao lado")
    args = ap.parse_args(argv)

    if args.relatorio:
        relatorio(args.relatorio, args.amostra, args.temas)
        return
    if not args.provider:
        demo()
        return
    if not (args.model and args.out):
        sys.exit("informe --model e --out")
    ras.KEY_ENV = args.key_env
    t = pd.read_csv(args.temas)
    ras.TEMAS = {int(r.id): (r.title, str(r.description)) for r in t.itertuples()}
    run(args.amostra, args.notas, args.provider, args.model, args.out, args.limit, args.max_tokens)


if __name__ == "__main__":
    main()
