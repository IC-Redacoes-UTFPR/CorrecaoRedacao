"""Tabela de comparacao: nossos modelos e a literatura, nas duas divisoes do essay-br.

- oficial: divisao aleatoria do essay-br (make_official_split.py), temas do teste vistos no treino
  (prompt-specific), a dos resultados publicados.
- cross-prompt: divisao por tema (make_cross_prompt_split.py), temas do teste nunca vistos.

Para cada resultado que existir em results/colab: QWK do total bruto, calibrado pela validacao
(deslocamento e quantil, ajustados nas notas da validacao e aplicados ao teste), Pearson e QWK por
competencia. Escreve results/colab/comparacao.md.

    python comparar.py
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from calibrate import _ler, fit_hist, fit_vies
from evaluate import evaluate_pair

R = "results/colab"


def val_escolhida(ft):
    """CSV de validacao da epoca escolhida numa rodada com epocas (epoca.txt gravado pelo notebook)."""
    try:
        k = re.search(r"epoca (\d+)", open(f"{R}/{ft}_lora/epoca.txt", encoding="utf-8").read()).group(1)
        return f"gptoss20b_{ft}_ep{k}ev_val"
    except (OSError, AttributeError):
        return None


RODADAS = [
    ("oficial", "BERTimbau base (regressao)", "bertimbau_oficial_test", "bertimbau_oficial_val"),
    ("oficial", "gpt-oss-20b LoRA, nota esperada", "gptoss20b_oficialev_test", val_escolhida("oficial")),
    ("cross-prompt", "gpt-oss-120b sem treino (mts_fs)", "gptoss120b_mts_fs_cp", "gptoss120b_mts_fs_val"),
    ("cross-prompt", "BERTimbau base (regressao)", "bertimbau_cp_test", "bertimbau_cp_val"),
    ("cross-prompt", "gpt-oss-20b LoRA 3 epocas, nota esperada", "gptoss20b_ft3ev_cp", "gptoss20b_ft3ev_val"),
    ("cross-prompt", "gpt-oss-20b LoRA ft6 (epoca pela validacao)", "gptoss20b_ft6ev_cp", val_escolhida("ft6")),
]

LITERATURA = [
    "| oficial | Amorim e Veloso (2017), features (Marinho et al., JIDM 2022) | 0,49 | | | | 0,39 | 0,46 | 0,40 | 0,38 | 0,34 |",
    "| oficial | Fonseca et al. (2018), features (Marinho et al., JIDM 2022) | 0,53 | | | | 0,44 | 0,48 | 0,42 | 0,47 | 0,38 |",
    "| aleatoria 70/15/15, semente nao informada | BERTimbau base (Matsuoka, arXiv 2023, sem revisao) | 0,79 | | | | 0,74 | 0,78 | 0,76 | 0,84 | 0,79 |",
]


def qwk_comps(path):
    df = pd.read_csv(path)
    out = []
    for c in ["c1", "c2", "c3", "c4", "c5"]:
        p = pd.to_numeric(df[f"pred_{c}"], errors="coerce")
        ok = p.notna()
        out.append(evaluate_pair(df.loc[ok, c].to_numpy(float), p[ok].to_numpy(float), is_total=False)["qwk"])
    return out


def f2(x):
    return f"{x:.2f}".replace(".", ",")


def linha(div, nome, teste, val):
    t = f"{R}/{teste}.csv"
    if not os.path.exists(t):
        return None
    gt, pt, _ = _ler(t)
    m = evaluate_pair(gt, pt, is_total=True)
    desloc = quantil = ""
    if val and os.path.exists(f"{R}/{val}.csv"):
        gv, pv, _ = _ler(f"{R}/{val}.csv")
        desloc = f2(evaluate_pair(gt, np.clip(fit_vies(pv, gv)(pt), 0, 1000), is_total=True)["qwk"])
        quantil = f2(evaluate_pair(gt, np.clip(fit_hist(pv, gv)(pt), 0, 1000), is_total=True)["qwk"])
    comps = " | ".join(f2(q) for q in qwk_comps(t))
    return f"| {div} | {nome} | {f2(m['qwk'])} | {desloc} | {quantil} | {f2(m['pearson'])} | {comps} |"


def main():
    cab = ["| Divisao | Modelo | QWK bruto | Calibrado pela val. (desloc.) | Calibrado pela val. (quantil) | Pearson | C1 | C2 | C3 | C4 | C5 |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    nossas = [l for l in (linha(*r) for r in RODADAS) if l]
    md = ("# Comparacao no essay-br estendido\n\nQWK do total no conjunto de teste; por competencia, QWK bruto.\n\n"
          "## Nossos modelos\n\n" + "\n".join(cab + nossas) +
          "\n\n## Literatura (mesmo corpus, divisao aleatoria: temas do teste vistos no treino)\n\n" +
          "\n".join(cab + LITERATURA) + "\n")
    os.makedirs(R, exist_ok=True)
    open(f"{R}/comparacao.md", "w", encoding="utf-8").write(md)
    print(md)


if __name__ == "__main__":
    main()
