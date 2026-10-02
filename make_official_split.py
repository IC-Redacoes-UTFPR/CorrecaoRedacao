"""Divisao oficial do essay-br estendido (prompt-specific), para comparar com a literatura.

Reproduz o build_dataset.py de github.com/lplnufpi/essay-br: train_test_split com
random_state=230, 70% treino e o resto dividido ao meio (validacao e teste). Apesar do nome da
funcao deles (split_stratified_into_train_val_test), ela nao passa `stratify`: a divisao e
aleatoria simples. Os temas do teste aparecem todos no treino (cenario prompt-specific, o dos
resultados publicados: Marinho et al., JIDM 2022; Matsuoka, arXiv 2401.00095).

O data/meu_dataset.csv e identico ao extended_essay-br.csv publicado (6.577 redacoes). As
redacoes de nota 0 ficam, como no original. index_redacao = linha no arquivo original.

Saida: data/oficial_train.csv (4.603), data/oficial_val.csv (987), data/oficial_test.csv (987).

    python make_official_split.py
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

COLS = ["index_redacao", "prompt", "title", "essay", "c1", "c2", "c3", "c4", "c5", "score"]


def main():
    df = pd.read_csv("data/meu_dataset.csv").reset_index(names="index_redacao")
    tr, tmp = train_test_split(df, test_size=0.3, random_state=230)
    va, te = train_test_split(tmp, test_size=0.5, random_state=230)
    for nome, p in (("train", tr), ("val", va), ("test", te)):
        p[COLS].to_csv(f"data/oficial_{nome}.csv", index=False)
        print(f"{nome:5s}: {len(p):5d} redacoes, {p['prompt'].nunique():3d} temas, media {p['score'].mean():.0f}")
    assert (len(tr), len(va), len(te)) == (4603, 987, 987)


if __name__ == "__main__":
    main()
