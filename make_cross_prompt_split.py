"""Divisao cross-prompt de verdade: treino, validacao e teste com TEMAS disjuntos.

O teste do v5 foi separado por nota e cobre 146 dos 151 temas, entao nao serve para avaliar
fine-tuning em cenario cross-prompt (o modelo treinado veria os temas do teste). Aqui os temas
sao sorteados por categoria (sociedade, saude, educacao...), para o teste ter a mesma mistura
de assuntos do banco.

Saida (mesmo formato da amostra_300): data/cp_train.csv, data/cp_val.csv, data/cp_test.csv,
com index_redacao = posicao no banco sem as redacoes de nota 0.

    python make_cross_prompt_split.py              # 20% dos temas no teste, 10% na validacao
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

SEED = 20261001
COLS = ["index_redacao", "prompt", "title", "essay", "c1", "c2", "c3", "c4", "c5", "score"]


def sortear_temas(temas, frac_teste, frac_val, seed=SEED):
    """Por categoria, sorteia os temas de teste e de validacao. Devolve (teste, val) como sets."""
    rng = np.random.default_rng(seed)
    teste, val = set(), set()
    for _, grp in temas.groupby("category"):
        ids = rng.permutation(grp["id"].to_numpy())
        nt = max(1, round(len(ids) * frac_teste))
        nv = max(1, round(len(ids) * frac_val))
        teste |= set(ids[:nt].tolist())
        val |= set(ids[nt:nt + nv].tolist())
    return teste, val


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="data/meu_dataset.csv")
    ap.add_argument("--temas", default="data/prompts_essaybr.csv")
    ap.add_argument("--frac-teste", type=float, default=0.2)
    ap.add_argument("--frac-val", type=float, default=0.1)
    args = ap.parse_args(argv)

    df = pd.read_csv(args.dataset)
    df = df[df["score"] > 0].reset_index(drop=True).reset_index(names="index_redacao")
    teste, val = sortear_temas(pd.read_csv(args.temas), args.frac_teste, args.frac_val)
    parte = np.where(df["prompt"].isin(teste), "test", np.where(df["prompt"].isin(val), "val", "train"))
    for nome in ("train", "val", "test"):
        p = df[parte == nome]
        p[COLS].to_csv(f"data/cp_{nome}.csv", index=False)
        print(f"{nome:5s}: {len(p):5d} redacoes, {p['prompt'].nunique():3d} temas, media {p['score'].mean():.0f}")
    assert not (set(df[parte == "test"]["prompt"]) & set(df[parte != "test"]["prompt"]))  # temas disjuntos


if __name__ == "__main__":
    main()
