"""Seleciona redacoes ancora para o modo few-shot (mts_fs) do run_api_scoring.py.

Base na ideia "anchor is the key" (Studies in Educational Evaluation, 2026): por
competencia, um exemplo por faixa de nota, tirado de FORA do conjunto de teste
(folds != 0 do v5), para o modelo calibrar a escala.

Saida: data/anchors.csv com colunas comp, band, score_comp, essay (truncada).

    python build_anchors.py                # 1 ancora por (competencia, faixa)
    python build_anchors.py --por-faixa 2 --max-chars 900
"""
from __future__ import annotations

import argparse

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from run_api_scoring import COMPS, limpar

BANDAS = [0, 40, 80, 120, 160, 200]


def train_df(dataset_path):
    df = pd.read_csv(dataset_path)
    df = df[df["score"] > 0].reset_index(drop=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    df["fold"] = -1
    for i, (_, te) in enumerate(skf.split(df, df["score"])):
        df.loc[te, "fold"] = i
    return df[df["fold"] != 0].reset_index(drop=True)  # tudo que NAO e teste


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="data/meu_dataset.csv")
    ap.add_argument("--treino", help="CSV de treino ja separado (ex data/cp_train.csv); ignora --dataset")
    ap.add_argument("--out", default="data/anchors.csv")
    ap.add_argument("--por-faixa", type=int, default=1, help="quantas ancoras por faixa")
    ap.add_argument("--max-chars", type=int, default=800, help="corte do texto da ancora")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    df = pd.read_csv(args.treino) if args.treino else train_df(args.dataset)
    df["essay_limpo"] = df["essay"].apply(limpar)
    # texto medio, nem o mais curto nem o mais longo, e de tamanho tratavel
    df["len"] = df["essay_limpo"].str.len()
    df = df[df["len"].between(600, 2500)]

    linhas = []
    for comp in COMPS:
        col = comp.lower()  # C1 -> c1
        for band in BANDAS:
            grp = df[df[col] == band].sort_values("len", key=lambda s: (s - s.median()).abs())
            for _, row in grp.head(args.por_faixa).iterrows():
                linhas.append({
                    "comp": comp, "band": band, "score_comp": band,
                    "essay": row["essay_limpo"][:args.max_chars],
                })
    out = pd.DataFrame(linhas)
    out.to_csv(args.out, index=False)

    print(f"{len(out)} ancoras -> {args.out}")
    print(out.groupby("comp")["band"].apply(list).to_string())


if __name__ == "__main__":
    main()
