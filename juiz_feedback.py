"""Juiz automatico do feedback por competencia (ideia do FeedEval, Chu et al., 2026).

Um LLM de outra familia (padrao: Gemini Flash Lite; o feedback foi gerado pelo gpt-oss-120b, e o
mesmo modelo tenderia a favorecer o proprio texto) le a redacao e o feedback das 5 competencias,
numa chamada por feedback, e da:
- por competencia, notas de 1 a 5 para especificidade (aponta trechos e problemas concretos),
  utilidade (o aluno sabe o que fazer para melhorar) e validade (o que diz sobre a redacao e
  verdade e segue a grade do ENEM);
- por problema apontado, se procede (e um problema real do trecho) e se esta na competencia certa.

Retoma de onde parou (pula index_redacao ja julgado no arquivo de saida).

    python juiz_feedback.py --feedback results/colab/feedback_cp30_comp3_ft3.jsonl \
        --amostra data/amostra_feedback_cp30.csv --out results/juiz/cp30_v3_ft3.jsonl
    python juiz_feedback.py --resumo results/juiz/*.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import pandas as pd

import run_api_scoring as ras
from run_api_scoring import COMPS, RUBRICA, chat, extrair_json, paragrafos


def bloco_feedback(fb):
    linhas = []
    for c in COMPS:
        f = fb.get(c) or {}
        linhas.append(f"### {c}\nPontos fortes: " + " | ".join(f.get("pontos_fortes") or []))
        for i, p in enumerate(f.get("problemas") or []):
            linhas.append(f"Problema {i}: trecho \"{p.get('trecho', '')}\". Explicacao: {p.get('explicacao', '')} "
                          f"Correcao sugerida: {p.get('correcao', '')}")
        linhas.append("Como melhorar: " + " | ".join(f.get("como_melhorar") or []))
    return "\n".join(linhas)


def prompt_juiz(essay, fb):
    grade = "\n".join(f"{c}: {RUBRICA[c]}" for c in COMPS)
    return (
        "Voce e um corretor experiente do ENEM avaliando a QUALIDADE de um feedback escrito para um "
        "estudante sobre a redacao abaixo. Nao avalie a redacao; avalie o feedback.\n\n"
        f"COMPETENCIAS DO ENEM:\n{grade}\n\n"
        f"REDACAO DO ESTUDANTE:\n{essay}\n\n"
        f"FEEDBACK A AVALIAR:\n{bloco_feedback(fb)}\n\n"
        "Para cada competencia, de notas de 1 a 5:\n"
        "- especificidade: 1 = generico, servia para qualquer redacao; 5 = aponta trechos e problemas "
        "concretos desta redacao.\n"
        "- utilidade: 1 = o estudante nao saberia o que fazer; 5 = orientacoes claras e acionaveis.\n"
        "- validade: 1 = afirmacoes falsas sobre a redacao ou contra a grade do ENEM; 5 = tudo correto.\n"
        "Para cada problema numerado da competencia, diga se procede (o trecho tem mesmo esse problema) "
        "e se competencia_certa (o problema pertence a essa competencia e nao a outra).\n"
        "Seja rigoroso: nota 5 so quando nao houver falha. Responda so com JSON neste formato:\n"
        '{"C1": {"especificidade": 4, "utilidade": 3, "validade": 5, '
        '"problemas": [{"i": 0, "procede": true, "competencia_certa": true}]}, "C2": {...}, ..., "C5": {...}}'
    )


def julgar(args):
    am = pd.read_csv(args.amostra).set_index("index_redacao")
    feitos = set()
    if os.path.exists(args.out):
        feitos = {json.loads(l)["index_redacao"] for l in open(args.out, encoding="utf-8")}
    regs = [json.loads(l) for l in open(args.feedback, encoding="utf-8")]
    if args.limit:
        regs = regs[: args.limit]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        for r in regs:
            if r["index_redacao"] in feitos:
                continue
            t0 = time.time()
            essay = paragrafos(am.loc[r["index_redacao"], "essay"])
            d = extrair_json(chat(args.provider, args.model, prompt_juiz(essay, r["feedback"]),
                                  temperature=0, max_tokens=4000))
            if not d or not all(c in d for c in COMPS):
                print(f"  {r['index_redacao']}: resposta invalida, pulando")
                continue
            f.write(json.dumps({"index_redacao": r["index_redacao"], "feedback": args.feedback,
                                "juiz": args.model, "notas": d}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{r['index_redacao']} ok ({time.time() - t0:.1f}s)")


def resumo(paths):
    linhas = []
    for p in paths:
        esp, uti, val, proc, cert = [], [], [], [], []
        for l in open(p, encoding="utf-8"):
            for c in COMPS:
                n = json.loads(l)["notas"][c]
                esp.append(n.get("especificidade"))
                uti.append(n.get("utilidade"))
                val.append(n.get("validade"))
                for q in n.get("problemas") or []:
                    proc.append(bool(q.get("procede")))
                    cert.append(bool(q.get("competencia_certa")))
        m = lambda x: sum(v for v in x if v is not None) / max(1, sum(v is not None for v in x))
        n_red = sum(1 for _ in open(p, encoding="utf-8"))
        linhas.append(f"| {os.path.basename(p)} | {n_red} | {m(esp):.2f} | {m(uti):.2f} | {m(val):.2f} | "
                      f"{100 * m(proc):.0f}% | {100 * m(cert):.0f}% |")
    cab = ("| Arquivo | Redacoes | Especificidade | Utilidade | Validade | Problemas procedentes | "
           "Na competencia certa |\n|---|---|---|---|---|---|---|")
    print(cab + "\n" + "\n".join(linhas))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feedback")
    ap.add_argument("--amostra")
    ap.add_argument("--out")
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--key-env", help="nome da variavel da chave (ex.: GEMINI_API_KEY_2)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resumo", nargs="+")
    args = ap.parse_args(argv)
    if args.resumo:
        return resumo(sorted(p for g in args.resumo for p in glob.glob(g)))
    ras.KEY_ENV = args.key_env
    julgar(args)


if __name__ == "__main__":
    main()
