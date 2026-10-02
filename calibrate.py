"""Experimento de recalibracao da nota prevista.

Pergunta: o QWK baixo vem de o modelo AVALIAR mal, ou de ele ORDENAR bem e
ESCALAR mal? Se for escala, um mapa simples aprendido em dados separados
recupera a maior parte do QWK, sem trocar de modelo.

Como funciona:
  - 5-fold. Agrupa por tema (coluna `prompt`, ou --prompts-file) quando possivel,
    para nao vazar tema entre treino e teste (cenario cross-prompt). Sem tema,
    cai para fold aleatorio por redacao e avisa que o numero fica otimista.
  - Para cada calibrador, ajusta no treino de cada fold e aplica no fold retido.
    Junta as predicoes retidas (out-of-fold) e mede tudo sobre o conjunto todo.
  - Tambem calcula a calibracao ORACULO (ajusta e mede nos mesmos dados): teto do
    que qualquer mapa monotonico consegue.

Calibradores:
  bruto            nada, e a linha de base
  vies             pred + (media_gold_treino - media_pred_treino)     1 parametro
  afim             a*pred + b por minimos quadrados                    2 parametros
  hist             casa o quantil de pred com o quantil das notas humanas
  isotonica        regressao isotonica (opcional, precisa de scikit-learn)

Leitura do resultado: mapa monotonico NAO muda Pearson nem Spearman. Se o QWK
sobe muito e o Pearson fica parado, o problema era escala, nao avaliacao.

Uso:
    python calibrate.py results/v5_experimentos_v2/ck1_qwen.csv \
        --out results/calibracao_ck1_qwen.csv --plot results/calibracao_ck1_qwen.png
    python calibrate.py            # roda o auto-teste
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from evaluate import evaluate_pair  # reaproveita o pacote de metricas

SEED = 42


# ---------------------------------------------------------------- calibradores
def fit_bruto(p, g):
    return lambda x: np.asarray(x, float)


def fit_vies(p, g):
    shift = float(np.mean(g) - np.mean(p))
    return lambda x: np.asarray(x, float) + shift


def fit_afim(p, g):
    a, b = np.polyfit(np.asarray(p, float), np.asarray(g, float), 1)
    return lambda x: a * np.asarray(x, float) + b


def fit_hist(p, g):
    ps, gs = np.sort(np.asarray(p, float)), np.sort(np.asarray(g, float))
    qp = np.linspace(0, 1, len(ps))
    qg = np.linspace(0, 1, len(gs))

    def f(x):
        u = np.interp(np.asarray(x, float), ps, qp)  # pred -> seu quantil
        return np.interp(u, qg, gs)                   # quantil -> nota humana

    return f


CALIBRADORES = {"bruto": fit_bruto, "vies": fit_vies, "afim": fit_afim, "hist": fit_hist}

try:  # isotonica so se o scikit-learn estiver instalado
    from sklearn.isotonic import IsotonicRegression

    def fit_isotonica(p, g):
        ir = IsotonicRegression(out_of_bounds="clip").fit(np.asarray(p, float), np.asarray(g, float))
        return lambda x: np.asarray(ir.predict(np.asarray(x, float)), float)

    CALIBRADORES["isotonica"] = fit_isotonica
except ImportError:
    pass


# ---------------------------------------------------------------- folds
def make_folds(n, groups, k=5, seed=SEED):
    """Indices de teste por fold. Agrupa por `groups` se dado, senao aleatorio."""
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.permutation(n)
        return [idx[i::k] for i in range(k)]
    uniq = np.array(sorted(pd.unique(groups)))
    rng.shuffle(uniq)
    bucket = {g: i % k for i, g in enumerate(uniq)}
    fold_of = np.array([bucket[g] for g in groups])
    return [np.where(fold_of == i)[0] for i in range(k)]


def out_of_fold(pred, gold, folds, fit):
    out = np.empty_like(pred, dtype=float)
    for test_idx in folds:
        mask = np.ones(len(pred), bool)
        mask[test_idx] = False
        f = fit(pred[mask], gold[mask])
        out[test_idx] = np.clip(f(pred[test_idx]), 0, 1000)
    return out


# ---------------------------------------------------------------- experimento
def _ler(path):
    df = pd.read_csv(path)
    gold = np.asarray(pd.to_numeric(df["score"], errors="coerce"), float)
    pred = np.asarray(pd.to_numeric(df["pred_total"], errors="coerce"), float)
    ok = ~(np.isnan(gold) | np.isnan(pred))
    return gold[ok], pred[ok], df.loc[ok].reset_index(drop=True)


def run(path, prompts_file=None, out_csv=None, plot_path=None, calib_file=None):
    gold, pred, df = _ler(path)

    groups = None
    if "prompt" in df.columns:
        groups = np.asarray(df["prompt"])
    elif prompts_file:
        pm = pd.read_csv(prompts_file)
        groups = np.asarray(df.merge(pm, on="index_redacao", how="left")["prompt"])
    if groups is None or pd.isna(groups).any():
        print("[aviso] sem tema por redacao: fold aleatorio, QWK pode ficar otimista "
              "para uso cross-prompt.", file=sys.stderr)
        groups = None

    folds = make_folds(len(gold), groups)
    rows = []
    for nome, fit in CALIBRADORES.items():
        oof = out_of_fold(pred, gold, folds, fit)
        orac = np.clip(fit(pred, gold)(pred), 0, 1000)  # ajusta e mede nos mesmos dados
        for tag, yp in (("out-of-fold", oof), ("oraculo", orac)):
            m = evaluate_pair(gold, yp, is_total=True)
            rows.append({"calibrador": nome, "tipo": tag,
                         "qwk": m["qwk"], "mae": m["mae"], "rmse": m["rmse"],
                         "pearson": m["pearson"], "spearman": m["spearman"],
                         "acc_adjacente": m["acc_adjacente"], "vies": m["vies"]})
        if calib_file:  # ajustado num conjunto separado (validacao) e aplicado aqui: o teste fica intocado
            gv, pv, _ = _ler(calib_file)
            m = evaluate_pair(gold, np.clip(fit(pv, gv)(pred), 0, 1000), is_total=True)
            rows.append({"calibrador": nome, "tipo": "validacao",
                         "qwk": m["qwk"], "mae": m["mae"], "rmse": m["rmse"],
                         "pearson": m["pearson"], "spearman": m["spearman"],
                         "acc_adjacente": m["acc_adjacente"], "vies": m["vies"]})

    res = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.float_format", lambda v: f"{v:.4f}"):
        print(f"\n{path}  (n={len(gold)})")
        print(res.to_string(index=False))
    if out_csv:
        res.to_csv(out_csv, index=False)
        print(f"salvo em {out_csv}")
    if plot_path:
        _plot(pred, gold, plot_path)
        print(f"grafico em {plot_path}")
    return res


def _plot(pred, gold, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = np.linspace(pred.min(), pred.max(), 200)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pred, gold, s=8, alpha=0.15, label="redacoes")
    ax.plot([0, 1000], [0, 1000], "k--", lw=1, label="y = x")
    ax.plot(xs, fit_afim(pred, gold)(xs), label="afim")
    ax.plot(xs, fit_hist(pred, gold)(xs), label="hist")
    ax.set_xlabel("nota prevista (bruta)")
    ax.set_ylabel("nota humana")
    ax.set_title("Mapa de calibracao pred -> nota humana")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- auto-teste
def demo():
    rng = np.random.default_rng(0)
    gold = rng.choice(np.arange(0, 1001, 40), size=600).astype(float)
    # modelo que ORDENA bem e ESCALA mal: comprime a faixa e desloca, ruido pequeno
    pred = 0.5 * gold + 100 + rng.normal(0, 25, size=gold.shape)

    folds = make_folds(len(gold), None)
    raw = evaluate_pair(gold, np.clip(pred, 0, 1000), is_total=True)
    aff = evaluate_pair(gold, out_of_fold(pred, gold, folds, fit_afim), is_total=True)

    assert aff["qwk"] > raw["qwk"] + 0.2, (raw["qwk"], aff["qwk"])
    assert abs(aff["pearson"] - raw["pearson"]) < 0.03, (raw["pearson"], aff["pearson"])
    assert abs(aff["vies"]) < abs(raw["vies"]), (raw["vies"], aff["vies"])
    print(f"demo ok: QWK {raw['qwk']:.3f} -> {aff['qwk']:.3f} com calibrador afim, "
          f"Pearson estavel ({raw['pearson']:.3f}), vies {raw['vies']:.0f} -> {aff['vies']:.0f}.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Experimento de recalibracao da nota prevista.")
    ap.add_argument("file", nargs="?", help="CSV de checkpoint com score e pred_total")
    ap.add_argument("--prompts-file", help="CSV com index_redacao,prompt para fold por tema")
    ap.add_argument("--out", help="CSV de saida com a tabela de metricas")
    ap.add_argument("--plot", help="PNG do grafico pred vs nota humana")
    ap.add_argument("--calib-file", help="CSV de predicao de outro conjunto (validacao): ajusta os "
                    "calibradores nele e aplica neste arquivo (linhas tipo 'validacao')")
    args = ap.parse_args(argv)
    if not args.file:
        demo()
        return
    run(args.file, args.prompts_file, args.out, args.plot, args.calib_file)


if __name__ == "__main__":
    main()
