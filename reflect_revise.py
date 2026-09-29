"""Reflect-and-Revise da rubrica de C5 (Automated Refinement of Essay Scoring Rubrics via
Reflect-and-Revise, CoNLL 2026), sem treino.

Loop, num fold de validacao com gabarito e fora do teste:
  1. o avaliador (modo mts_rr, so C5) pontua o fold com a rubrica da versao k;
  2. o mesmo modelo le a rubrica, as estatisticas e os casos com erro >= 80 pontos
     (nota humana, nota dele, justificativa, trecho final da redacao) e reescreve a rubrica;
  3. repete. Escolhe a versao com maior QWK de C5 na validacao e so ela vai para o teste.

Fold de validacao: 100 redacoes dos folds de treino do v5, de temas que NAO aparecem na
amostra de teste (cross-prompt), sem as redacoes ancora, com todas as faixas de C5.

Uso:
    python reflect_revise.py --build-val                      # cria data/val_c5.csv
    python reflect_revise.py --iters 1 --limit 5              # smoke test (~11 chamadas)
    python reflect_revise.py --iters 5 --keys 1,2,3,4,5       # valendo (~605 chamadas)
    python reflect_revise.py --merge-test                     # junta rrtest_*.csv no mts_fs2

Saidas em results/rr/: val_v{k}.csv, historico.csv. Rubricas em data/rubricas_c5/v{k}.txt,
com o diagnostico do modelo em v{k}_diagnostico.txt.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

import run_api_scoring as r
from build_anchors import train_df
from evaluate import evaluate_pair

VAL = "data/val_c5.csv"
RUB_DIR = "data/rubricas_c5"
OUT_DIR = "results/rr"
BANDAS = [0, 40, 80, 120, 160, 200]


def build_val(dataset="data/meu_dataset.csv", n=100, seed=20260929):
    tr = train_df(dataset).reset_index(names="index_redacao")
    teste = pd.read_csv("data/amostra_300.csv")
    ancoras = set(pd.read_csv("data/anchors.csv")["essay"])
    tr = tr[~tr["prompt"].isin(teste["prompt"])]
    tr = tr[~tr["essay"].map(lambda e: r.limpar(e)[:800] in ancoras)]
    # proporcional a C5, com piso de 6 por faixa para toda faixa ter erro para refletir
    cota = (tr["c5"].value_counts(normalize=True) * n).round().clip(lower=6).astype(int)
    cota = (cota * n / cota.sum()).round().astype(int)
    val = pd.concat(g.sample(min(cota[b], len(g)), random_state=seed) for b, g in tr.groupby("c5"))
    cols = ["index_redacao", "prompt", "title", "essay", "c1", "c2", "c3", "c4", "c5", "score"]
    val[cols].sort_values("index_redacao").to_csv(VAL, index=False)
    print(f"{len(val)} redacoes -> {VAL}, {val['prompt'].nunique()} temas fora do teste")
    print(val["c5"].value_counts().sort_index().to_string())


def metricas(out):
    d = pd.read_csv(out)
    d = d[d["ok"] == 1]
    m = evaluate_pair(d["c5"], d["pred_c5"], is_total=False)
    return d, m


def prompt_revisao(rubrica, d, val):
    d = d.merge(val[["index_redacao", "essay"]], on="index_redacao")
    d["erro"] = d["pred_c5"] - d["c5"]
    ruins = d[d["erro"].abs() >= 80]
    casos = pd.concat([ruins.nlargest(6, "erro"), ruins.nsmallest(6, "erro")]).drop_duplicates("index_redacao")
    conf = pd.crosstab(d["c5"], d["pred_c5"]).to_string()
    blocos = "\n\n".join(
        f"CASO {i + 1}: humano {int(c.c5)}, avaliador {int(c.pred_c5)}\n"
        f"Justificativa do avaliador: {str(c.raw).removeprefix('C5: ')}\n"
        f"Trecho final da redacao: ...{r.limpar(c.essay)[-900:]}"
        for i, c in enumerate(casos.itertuples()))
    return (
        "Voce esta refinando a rubrica que um avaliador automatico usa para pontuar a "
        "competencia 5 do ENEM (proposta de intervencao), em 0, 40, 80, 120, 160 ou 200.\n\n"
        f"RUBRICA ATUAL:\n{rubrica}\n\n"
        f"DESEMPENHO NA VALIDACAO ({len(d)} redacoes): media humana {d['c5'].mean():.0f}, "
        f"media do avaliador {d['pred_c5'].mean():.0f}; "
        f"{(d['erro'] > 0).sum()} notas acima do humano, {(d['erro'] < 0).sum()} abaixo.\n"
        f"Matriz (linhas = humano, colunas = avaliador):\n{conf}\n\n"
        f"CASOS COM ERRO DE 80 PONTOS OU MAIS:\n\n{blocos}\n\n"
        "Tarefa: (1) identifique os padroes SISTEMATICOS de erro, nao casos isolados; "
        "(2) reescreva a rubrica para corrigi-los, mantendo o que ja funciona.\n"
        "Regras: rubrica geral, sem citar temas nem redacoes especificas; no maximo 250 "
        "palavras; escala 0, 40, 80, 120, 160, 200; o avaliador vai receber exemplos ancora "
        "depois da rubrica e responder JSON com C5, elementos e justificativa, entao nao "
        "mude esse formato de resposta.\n\n"
        'Responda APENAS com JSON: {"diagnostico": "2 a 4 frases", "rubrica": "texto novo"}'
    )


def revisar(model, rubrica, d, val):
    resp = r.chat("gemini", model, prompt_revisao(rubrica, d, val), temperature=0.4, max_tokens=3000)
    t = re.sub(r"```json|```", "", resp)
    j = json.loads(t[t.find("{"): t.rfind("}") + 1])
    return j["rubrica"].strip(), j.get("diagnostico", "")


def loop(model, iters, keys, limit, rpm):
    os.makedirs(RUB_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    val = pd.read_csv(VAL)
    v0 = f"{RUB_DIR}/v0.txt"
    if not os.path.exists(v0):
        with open(v0, "w", encoding="utf-8") as fh:
            fh.write(r.C5_RUBRICA_V0)
    r.PROVIDERS["gemini"]["rpm"] = rpm
    hist = []
    for k in range(iters + 1):
        path = f"{RUB_DIR}/v{k}.txt"
        r.KEY_ENV = f"GEMINI_API_KEY_{keys[k % len(keys)]}"
        r.RUBRICA_C5_PATH = path
        out = f"{OUT_DIR}/val_v{k}.csv"
        print(f"\n=== v{k} ({r.KEY_ENV}) ===")
        r.run(VAL, "gemini", model, "mts_rr", out, 0.1, limit, 0, ["C5"])
        d, m = metricas(out)
        hist.append({"versao": k, "n": m["n"], "qwk_c5": round(m["qwk"], 3),
                     "vies": round(m["vies"], 1), "mae": round(m["mae"], 1)})
        print(f"v{k}: QWK C5 {m['qwk']:.3f}, vies {m['vies']:+.0f}, n {m['n']}")
        pd.DataFrame(hist).to_csv(f"{OUT_DIR}/historico.csv", index=False)
        if k == iters or os.path.exists(f"{RUB_DIR}/v{k + 1}.txt"):
            continue  # retomada: nao reescreve rubrica que ja existe
        with open(path, encoding="utf-8") as fh:
            nova, diag = revisar(model, fh.read().strip(), d, val)
        with open(f"{RUB_DIR}/v{k + 1}.txt", "w", encoding="utf-8") as fh:
            fh.write(nova)
        with open(f"{RUB_DIR}/v{k + 1}_diagnostico.txt", "w", encoding="utf-8") as fh:
            fh.write(diag)
        print(f"diagnostico: {diag}")
    h = pd.DataFrame(hist)
    best = int(h.loc[h["qwk_c5"].idxmax(), "versao"])
    print(f"\n{h.to_string(index=False)}\n\nmelhor: v{best} -> {RUB_DIR}/v{best}.txt")


def merge_test(trocas=None, out="results/api/flashlite_mtsrr.csv", modo="mts_rr",
               base="results/api/flashlite_mtsfs2.csv"):
    """Troca competencias do mts_fs2 pelas de rodadas de uma competencia so (--comps).
    trocas: {"C5": "results/api/rrtest_*.csv", ...}. As outras competencias usam o mesmo
    prompt nos dois modos, entao a troca equivale a rodar o modo novo inteiro."""
    trocas = trocas or {"C5": "results/api/rrtest_*.csv"}
    d = pd.read_csv(base)
    partes = {c: str(t).split(" || ") for c, t in zip(d["index_redacao"], d["raw"])}
    for comp, padrao in trocas.items():
        novo = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(padrao)))
        novo = novo[novo["ok"] == 1].drop_duplicates("index_redacao").set_index("index_redacao")
        d = d[d["index_redacao"].isin(novo.index)].copy()
        col = f"pred_{comp.lower()}"
        d[col] = d["index_redacao"].map(novo[col])
        for i, raw in novo["raw"].astype(str).items():
            partes[i] = [p for p in partes.get(i, []) if not p.startswith(comp + ":")] + [raw]
    d["pred_total"] = d[[f"pred_c{i}" for i in range(1, 6)]].sum(axis=1)
    d["raw"] = d["index_redacao"].map(lambda i: " || ".join(sorted(partes[i]))[:1500])
    d["modo"] = modo
    d.to_csv(out, index=False)
    print(f"{len(d)} redacoes -> {out} (trocadas: {', '.join(trocas)})")


def demo():
    val = pd.DataFrame({"index_redacao": [1, 2, 3], "essay": ["a " * 10, "b " * 10, "c " * 10]})
    d = pd.DataFrame({"index_redacao": [1, 2, 3], "c5": [0, 120, 200], "pred_c5": [160, 120, 40],
                      "raw": ["C5: sem proposta", "C5: ok", "C5: parcial"]})
    p = prompt_revisao("RUB", d, val)
    assert "humano 0, avaliador 160" in p and "humano 200, avaliador 40" in p
    assert "humano 120" not in p  # acerto nao entra como caso
    print("demo ok: casos de erro >= 80 selecionados, acertos fora do prompt de revisao.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-val", action="store_true")
    ap.add_argument("--merge-test", action="store_true")
    ap.add_argument("--merge", nargs="+", metavar="COMP=GLOB",
                    help="ex C1=results/api/lttest_*.csv C5=results/api/rrtest_*.csv")
    ap.add_argument("--merge-out", default="results/api/flashlite_mtsrr.csv")
    ap.add_argument("--merge-modo", default="mts_rr")
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--keys", default="1", help="numeros das contas, ex 1,2,3,4,5 (uma por versao)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rpm", type=int, default=15)
    args = ap.parse_args(argv)
    if args.build_val:
        build_val()
    elif args.merge_test or args.merge:
        trocas = dict(m.split("=", 1) for m in args.merge) if args.merge else None
        merge_test(trocas, args.merge_out, args.merge_modo)
    elif not os.path.exists(VAL):
        demo()
        print(f"{VAL} nao existe, rode --build-val")
    else:
        loop(args.model, args.iters, args.keys.split(","), args.limit, args.rpm)


if __name__ == "__main__":
    main()
