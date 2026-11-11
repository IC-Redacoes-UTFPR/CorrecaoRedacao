"""Empilhamento por competencia na divisao cross-prompt (so CPU, com as previsoes ja salvas).

Para cada competencia, uma regressao ridge sobre as notas previstas pelo gpt-oss-20b LoRA (ft3,
nota esperada) e pelo BERTimbau base, mais features simples do texto (palavras, paragrafos,
diversidade de vocabulario, tamanho medio da frase), na linha do EASY (Choi et al., 2026).
Ajuste so na validacao (o treino nao tem previsoes fora da amostra). A estimativa na validacao e
out-of-fold por tema (GroupKFold); o teste e medido uma vez com o modelo ajustado na validacao
inteira. Calibracao por quantil ajustada nas previsoes out-of-fold da validacao.
A ultima variante soma os desvios do LanguageTool (data/lt_cp_val.csv e data/lt_cp_test.csv).

    python empilhar.py
"""
from __future__ import annotations

import ast

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from calibrate import fit_hist
from evaluate import evaluate_pair

R = "results/colab"
C = ["c1", "c2", "c3", "c4", "c5"]
MODELOS = {"ft3": "gptoss20b_ft3ev", "bert": "bertimbau_cp"}


def feats_texto(essay):
    try:
        pars = [p for p in ast.literal_eval(essay) if p.strip()]
    except (ValueError, SyntaxError):
        pars = [essay]
    txt = " ".join(pars)
    pal = txt.split()
    frases = max(1, sum(txt.count(s) for s in ".!?"))
    return {"n_pal": len(pal), "n_par": len(pars),
            "ttr": len({p.lower() for p in pal}) / max(1, len(pal)), "pal_frase": len(pal) / frases}


def carregar(div):
    """div: 'val' ou 'test'. Junta as previsoes dos dois modelos com o texto e o tema."""
    base = pd.read_csv(f"data/cp_{div}.csv")[["index_redacao", "prompt", "essay"] + C + ["score"]]
    df = base.join(pd.DataFrame([feats_texto(e) for e in base.essay]))
    for m, pref in MODELOS.items():
        nome = f"{pref}_val" if div == "val" else (f"{pref}_cp" if m == "ft3" else f"{pref}_test")
        p = pd.read_csv(f"{R}/{nome}.csv")[["index_redacao"] + [f"pred_{c}" for c in C]]
        df = df.merge(p.rename(columns={f"pred_{c}": f"{m}_{c}" for c in C}), on="index_redacao")
    # desvios do LanguageTool por 100 palavras (lt_features.py --amostra data/cp_{div}.csv)
    lt = pd.read_csv(f"data/lt_cp_{div}.csv")
    for k in LT[1:]:
        lt[k] = 100 * lt[k] / lt.palavras
    df = df.merge(lt[["index_redacao"] + LT], on="index_redacao")
    return df.dropna(subset=[f"{m}_{c}" for m in MODELOS for c in C]).reset_index(drop=True)


LT = ["desvios_100", "ortografia", "gramatica", "pontuacao"]
COLS = [f"{m}_{c}" for m in MODELOS for c in C] + ["n_pal", "n_par", "ttr", "pal_frase"]


def novo():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))


def prever(tr, te, cols):
    out = {}
    for c in C:
        mod = novo().fit(tr[cols], tr[c])
        out[c] = np.clip(mod.predict(te[cols]), 0, 200)
    return pd.DataFrame(out)


def oof(val, cols):
    pred = pd.DataFrame(index=val.index, columns=C, dtype=float)
    for tr, te in GroupKFold(n_splits=5).split(val, groups=val.prompt):
        pred.iloc[te] = prever(val.iloc[tr], val.iloc[te], cols).to_numpy()
    return pred


def metr(df, pred, calib=None):
    tot = pred[C].sum(axis=1).to_numpy(float)
    if calib is not None:
        tot = np.clip(calib(tot), 0, 1000)
    m = evaluate_pair(df.score.to_numpy(float), tot, is_total=True)
    comps = [evaluate_pair(df[c].to_numpy(float), pred[c].to_numpy(float), is_total=False)["qwk"] for c in C]
    return m["qwk"], m["pearson"], comps


def f2(x):
    return f"{x:.2f}".replace(".", ",")


def main():
    val, test = carregar("val"), carregar("test")
    print(f"validacao {len(val)} redacoes ({val.prompt.nunique()} temas), teste {len(test)}")
    so = lambda d, m: pd.DataFrame({c: d[f"{m}_{c}"] for c in C})
    media = lambda d: pd.DataFrame({c: (d[f"ft3_{c}"] + d[f"bert_{c}"]) / 2 for c in C})
    variantes = {
        "ft3 sozinho": (so(val, "ft3"), lambda d: so(d, "ft3")),
        "BERTimbau sozinho": (so(val, "bert"), lambda d: so(d, "bert")),
        "media simples": (media(val), media),
        "ridge, so os dois modelos": "modelos",
        "ridge, modelos + texto": "tudo",
        "ridge, modelos + texto + LanguageTool": "lt",
    }
    cab = "| Variante | Val. OOF QWK | Val. OOF Pearson | Teste QWK bruto | Teste QWK quantil | Teste Pearson | C1 | C2 | C3 | C4 | C5 |"
    linhas = [cab, "|" + "---|" * 11]
    for nome, v in variantes.items():
        if isinstance(v, str):
            cols = {"modelos": COLS[:10], "tudo": COLS, "lt": COLS + LT}[v]
            pv, pt = oof(val, cols), prever(val, test, cols)
        else:
            pv, pt = v[0], v[1](test)
        q_v, r_v, _ = metr(val, pv)
        cal = fit_hist(pv[C].sum(axis=1).to_numpy(float), val.score.to_numpy(float))
        q_t, r_t, comps = metr(test, pt)
        q_h, _, _ = metr(test, pt, cal)
        linhas.append(f"| {nome} | {f2(q_v)} | {f2(r_v)} | {f2(q_t)} | {f2(q_h)} | {f2(r_t)} | "
                      + " | ".join(f2(x) for x in comps) + " |")
    md = "\n".join(linhas)
    open(f"{R}/empilhamento.md", "w", encoding="utf-8").write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
