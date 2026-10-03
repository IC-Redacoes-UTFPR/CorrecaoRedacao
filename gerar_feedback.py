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
import ast
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


def paragrafos(texto):
    """O essay-br guarda a redacao como lista de paragrafos ("['p1', 'p2']") e o limpar junta tudo
    numa linha so. No feedback os paragrafos ficam: o modelo precisa achar a conclusao (C5) e nao
    deve citar um trecho que atravessa dois paragrafos."""
    try:
        ps = ast.literal_eval(str(texto))
    except (ValueError, SyntaxError):
        return limpar(texto)
    if not isinstance(ps, list):
        return limpar(texto)
    return "\n\n".join(p for p in map(limpar, ps) if p)


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
        "versao melhorada ou \"\"). Na correcao, nao invente dados, numeros, prazos ou "
        "porcentagens que nao estao na redacao.\n"
        "- como_melhorar: 1 a 3 acoes concretas para a proxima redacao, ligadas aos problemas.\n"
        "Use trechos diferentes em cada competencia, cada um mostrando um problema daquela "
        "competencia. O tom e a gravidade devem acompanhar a nota: nota alta, problemas "
        "pequenos; com nota 200, problemas pode ser lista vazia.\n\n"
        "Responda APENAS com JSON: {" + ", ".join(f'"{c}": {item}' for c in COMPS) + "}"
    )


# Niveis oficiais de cada competencia (Cartilha do Participante, INEP). Dizem ao modelo o que a
# nota significa, para o feedback seguir a nota e nao o julgamento proprio do modelo.
NIVEIS_ENEM = {
    "C1": {200: "excelente dominio da modalidade escrita formal e de escolha de registro; desvios so como excepcionalidade e sem reincidencia",
           160: "bom dominio da modalidade escrita formal e de escolha de registro, com poucos desvios gramaticais e de convencoes da escrita",
           120: "dominio mediano da modalidade escrita formal e de escolha de registro, com alguns desvios gramaticais e de convencoes da escrita",
           80: "dominio insuficiente da modalidade escrita formal, com muitos desvios gramaticais, de escolha de registro e de convencoes da escrita",
           40: "dominio precario da modalidade escrita formal, de forma sistematica, com diversificados e frequentes desvios gramaticais, de escolha de registro e de convencoes da escrita",
           0: "desconhecimento da modalidade escrita formal da lingua portuguesa"},
    "C2": {200: "desenvolve o tema por meio de argumentacao consistente, a partir de repertorio sociocultural produtivo, e apresenta excelente dominio do texto dissertativo-argumentativo",
           160: "desenvolve o tema por meio de argumentacao consistente e apresenta bom dominio do texto dissertativo-argumentativo, com proposicao, argumentacao e conclusao",
           120: "desenvolve o tema por meio de argumentacao previsivel e apresenta dominio mediano do texto dissertativo-argumentativo, com proposicao, argumentacao e conclusao",
           80: "desenvolve o tema recorrendo a copia de trechos dos textos motivadores ou apresenta dominio insuficiente do texto dissertativo-argumentativo, sem atender a estrutura com proposicao, argumentacao e conclusao",
           40: "apresenta o assunto, tangenciando o tema, ou demonstra dominio precario do texto dissertativo-argumentativo, com tracos constantes de outros tipos textuais",
           0: "fuga ao tema ou nao atendimento a estrutura dissertativo-argumentativa"},
    "C3": {200: "apresenta informacoes, fatos e opinioes relacionados ao tema, de forma consistente e organizada, configurando autoria, em defesa de um ponto de vista",
           160: "apresenta informacoes, fatos e opinioes relacionados ao tema, de forma organizada, com indicios de autoria, em defesa de um ponto de vista",
           120: "apresenta informacoes, fatos e opinioes relacionados ao tema, limitados aos argumentos dos textos motivadores e pouco organizados, em defesa de um ponto de vista",
           80: "apresenta informacoes, fatos e opinioes relacionados ao tema, mas desorganizados ou contraditorios e limitados aos argumentos dos textos motivadores",
           40: "apresenta informacoes, fatos e opinioes pouco relacionados ao tema ou incoerentes e sem defesa de um ponto de vista",
           0: "apresenta informacoes, fatos e opinioes nao relacionados ao tema e sem defesa de um ponto de vista"},
    "C4": {200: "articula bem as partes do texto e apresenta repertorio diversificado de recursos coesivos",
           160: "articula as partes do texto com poucas inadequacoes e apresenta repertorio diversificado de recursos coesivos",
           120: "articula as partes do texto, de forma mediana, com inadequacoes, e apresenta repertorio pouco diversificado de recursos coesivos",
           80: "articula as partes do texto, de forma insuficiente, com muitas inadequacoes, e apresenta repertorio limitado de recursos coesivos",
           40: "articula as partes do texto de forma precaria",
           0: "nao articula as informacoes"},
    "C5": {200: "elabora muito bem proposta de intervencao, detalhada, relacionada ao tema e articulada a discussao desenvolvida no texto",
           160: "elabora bem proposta de intervencao relacionada ao tema e articulada a discussao desenvolvida no texto",
           120: "elabora, de forma mediana, proposta de intervencao relacionada ao tema e articulada a discussao desenvolvida no texto",
           80: "elabora, de forma insuficiente, proposta de intervencao relacionada ao tema, ou nao articulada com a discussao desenvolvida no texto",
           40: "apresenta proposta de intervencao vaga, precaria ou relacionada apenas ao assunto",
           0: "nao apresenta proposta de intervencao ou apresenta proposta nao relacionada ao tema ou ao assunto"},
}


def banda(x):
    """Nota continua (ex. nota esperada do modelo treinado) -> faixa do ENEM mais proxima."""
    return int(min(200, max(0, round(float(x) / 40) * 40)))


def prompt_feedback_comp(essay, comp, nota, tema_id=None, ja_citados=()):
    """Feedback de UMA competencia, guiado pelo nivel oficial da nota e pelo nivel de cima."""
    tema = f"TEMA DA REDACAO: {ras.TEMAS[int(tema_id)][0]}\n\n" if ras.TEMAS and tema_id is not None else ""
    acima = (f"Para chegar a {nota + 40}, o nivel pede: {NIVEIS_ENEM[comp][nota + 40]}.\n" if nota < 200
             else "Esta e a nota maxima.\n")
    citados = ("Trechos ja citados nas outras competencias (escolha outros): "
               + " | ".join(f'"{t}"' for t in ja_citados) + "\n" if ja_citados else "")
    c5 = "Na C5, cite trechos da proposta de intervencao, que em geral fica no ultimo paragrafo.\n" if comp == "C5" else ""
    # na v2 a C2 e a C3 apontavam erros de gramatica: era quase toda a repeticao que sobrava
    so_c1 = ("Desvios gramaticais (ortografia, acentuacao, concordancia, regencia, crase, pontuacao) "
             "sao avaliados na C1: nao os aponte aqui, nem na explicacao.\n" if comp != "C1" else "")
    return (
        "Voce e professor de redacao do ENEM e vai escrever um feedback formativo para o "
        f"estudante, so sobre a competencia {comp}: {RUBRICA[comp]}\n\n"
        f"{tema}REDACAO:\n{essay}\n\n"
        f"A redacao ja foi corrigida e tirou {nota} de 200 em {comp}. A nota e final: nao a reavalie.\n"
        f"Nivel oficial do ENEM para {nota}: {NIVEIS_ENEM[comp][nota]}.\n{acima}\n"
        "Escreva, falando diretamente com o estudante (voce), em portugues:\n"
        "- pontos_fortes: 1 ou 2 coisas que justificam a nota nao ser menor.\n"
        "- problemas: o que mantem a redacao neste nivel e nao no de cima. A quantidade e a "
        "gravidade devem corresponder ao nivel: perto de 200, 0 a 1 problema leve; no meio, 1 a 2; "
        "nas notas baixas, ate 3 graves. Em \"trecho\", copie LITERALMENTE so a parte da redacao "
        "que mostra o problema (de 3 a 15 palavras, dentro de um paragrafo), com os erros "
        "exatamente como estao, sem corrigir nada. Em \"explicacao\", diga o "
        "que esta errado nos termos desta competencia. Em \"correcao\", reescreva o trecho do jeito "
        "certo (obrigatorio na C1), sem inventar dados, numeros ou porcentagens.\n"
        f"{so_c1}{c5}{citados}"
        "- como_melhorar: 1 a 3 acoes concretas para chegar ao nivel de cima.\n\n"
        'Responda APENAS com JSON: {"pontos_fortes": ["..."], "problemas": [{"trecho": "...", '
        '"explicacao": "...", "correcao": "..."}], "como_melhorar": ["..."]}'
    )


def alinhamento(registros):
    """O feedback acompanha a nota? Media de problemas por nota e Spearman (nota x n de problemas):
    quanto mais negativo, mais o numero de problemas cai quando a nota sobe."""
    pares = pd.DataFrame([(r["notas"][c], len((r["feedback"].get(c) or {}).get("problemas") or []))
                          for r in registros for c in COMPS], columns=["nota", "problemas"])
    media = pares.groupby("nota")["problemas"].mean().round(1).to_dict()
    return media, pares["nota"].corr(pares["problemas"], method="spearman")


def repetidos(registros):
    """Problemas que repetem um trecho ja citado na mesma redacao: um contem o outro e o menor tem
    pelo menos metade do tamanho do maior. Trecho curto dentro de um longo nao conta (costuma ser
    outro problema, ex. um erro de C1 dentro da frase da proposta)."""
    rep = tot = 0
    for r in registros:
        vistos = []
        for c in COMPS:
            for p in (r["feedback"].get(c) or {}).get("problemas") or []:
                t = norm(p.get("trecho", ""))
                rep += any(t and v and (t in v or v in t) and 2 * min(len(t), len(v)) >= max(len(t), len(v))
                           for v in vistos)
                tot += 1
                vistos.append(t)
    return rep, tot


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
    return {int(i): {c: banda(p.loc[i, f"pred_{c.lower()}"]) for c in COMPS}
            for i in df["index_redacao"] if i in p.index and p.loc[i, [f"pred_{c.lower()}" for c in COMPS]].notna().all()}


def feedback_por_competencia(provider, model, essay, notas, tema_id, max_tokens):
    """Uma chamada por competencia, em ordem, passando os trechos ja citados para nao repetir."""
    fb, citados = {}, []
    for c in COMPS:
        # a C5 nao recebe a lista: a proposta costuma ja ter sido citada (ex. erro de C1 nela), e
        # com "escolha outros" a C5 criticava a introducao por nao ter proposta (redacao 547)
        d = extrair_json(chat(provider, model, prompt_feedback_comp(essay, c, notas[c], tema_id,
                                                                    citados if c != "C5" else ()),
                              0.1, max_tokens))
        if not isinstance(d.get("problemas", []), list):
            return {}
        fb[c] = d
        citados += [p.get("trecho", "") for p in d.get("problemas") or [] if isinstance(p, dict)]
    return fb


def run(amostra, fonte, provider, model, out, limit=0, max_tokens=6000, por_comp=False):
    df = pd.read_csv(amostra)
    df = df.head(limit) if limit else df
    notas = notas_de(fonte, df)
    formato = "por_competencia" if por_comp else "junto"
    feitos = set()
    if os.path.exists(out):
        prev = [json.loads(l) for l in open(out, encoding="utf-8")]
        outros = [r for r in prev if (r["modelo"], r["notas_fonte"], r.get("formato", "junto")) != (model, fonte, formato)]
        if outros:  # mesmo cuidado do run_api_scoring: nao misturar rodadas diferentes
            sys.exit(f"{out} tem linhas de outro modelo/fonte de nota/formato ({outros[0]['modelo']}, "
                     f"{outros[0]['notas_fonte']}, {outros[0].get('formato', 'junto')}). Use outro --out.")
        feitos = {r["index_redacao"] for r in prev}
        print(f"retomando: {len(feitos)} ja feitas")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    lit_tot = [0, 0]
    for row in df.itertuples():
        idx = int(row.index_redacao)
        if idx in feitos or idx not in notas:
            continue
        essay = paragrafos(row.essay)
        try:
            if por_comp:
                fb = feedback_por_competencia(provider, model, essay, notas[idx], row.prompt, max_tokens)
            else:
                fb = extrair_json(chat(provider, model, prompt_feedback(essay, notas[idx], row.prompt), 0.1, max_tokens))
        except RuntimeError as e:
            print(f"[{idx}] {e}, pulando", file=sys.stderr)
            continue
        if not all(isinstance(fb.get(c), dict) for c in COMPS):
            print(f"[{idx}] FALHA parse")
            continue
        lit, tot = marcar_literal(fb, essay)
        lit_tot = [lit_tot[0] + lit, lit_tot[1] + tot]
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"index_redacao": idx, "modelo": model, "notas_fonte": fonte, "formato": formato,
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
    registros = [json.loads(l) for l in open(jsonl, encoding="utf-8")]
    for r in registros:
        i, a = r["index_redacao"], am.loc[r["index_redacao"]]
        linhas += [f"\n---\n\n## Redacao {i}\n", f"**Tema:** {temas.get(a['prompt'], '?')}\n",
                   "| | C1 | C2 | C3 | C4 | C5 | Total |", "|---|---|---|---|---|---|---|",
                   "| Humano | " + " | ".join(str(a[c.lower()]) for c in COMPS) + f" | {a['score']} |",
                   f"| Usada no feedback ({r['notas_fonte'] if r['notas_fonte'] == 'humano' else 'modelo'}) | "
                   + " | ".join(str(r["notas"][c]) for c in COMPS) + f" | {sum(r['notas'].values())} |\n",
                   "**Texto:**\n\n> " + paragrafos(a["essay"]).replace("\n\n", "\n>\n> ") + "\n"]
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
    media, rho = alinhamento(registros)
    rep, _ = repetidos(registros)
    resumo = (f"Trechos literais: {lit}/{tot}. Trechos repetidos entre competencias: {rep}/{tot}. "
              f"Problemas apontados por nota: {media}. "
              f"Spearman nota x numero de problemas: {rho:.2f} (quanto mais negativo, mais o feedback acompanha a nota).\n")
    linhas.insert(2, resumo)
    md = os.path.splitext(jsonl)[0] + ".md"
    open(md, "w", encoding="utf-8").write("\n".join(linhas))
    print(f"relatorio em {md}\n{resumo}")


def demo():
    essay = "O governo deve agir, pois ninguem aceitavam isso na sociedade."
    fb = {c: {"pontos_fortes": [], "problemas": [], "como_melhorar": []} for c in COMPS}
    fb["C1"]["problemas"] = [{"trecho": "ninguem aceitavam isso", "explicacao": "concordancia"},
                             {"trecho": "ninguem aceitava isso", "explicacao": "copia corrigida"}]
    assert marcar_literal(fb, essay) == (1, 2)
    assert fb["C1"]["problemas"][1]["literal"] is False  # corrigir ao copiar conta como nao literal
    assert extrair_json('```json\n{"C1": {"problemas": []}}\n```')["C1"] == {"problemas": []}
    assert paragrafos("['Um  texto.', 'Dois.']") == "Um texto.\n\nDois."
    assert paragrafos("sem lista") == "sem lista"
    fb["C4"]["problemas"] = [{"trecho": "Ninguem aceitavam isso!"}, {"trecho": "aceitavam"}]
    # o 1o da C4 repete o 1o da C1; "aceitavam" esta dentro dele, mas curto: outro problema
    assert repetidos([{"feedback": fb}]) == (1, 4)
    assert "avaliados na C1" in prompt_feedback_comp(essay, "C2", 120)
    assert "avaliados na C1" not in prompt_feedback_comp(essay, "C1", 120)
    print("demo ok: literalidade, paragrafos, trechos repetidos e gramatica so na C1.")


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
    ap.add_argument("--por-competencia", action="store_true",
                    help="uma chamada por competencia, guiada pelo nivel oficial da nota e pelo nivel de cima")
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
    run(args.amostra, args.notas, args.provider, args.model, args.out, args.limit, args.max_tokens,
        args.por_competencia)


if __name__ == "__main__":
    main()
