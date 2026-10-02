"""Baseline BERTimbau: regressao das 5 notas, para comparar com a literatura do essay-br.

Como em Matsuoka (arXiv 2401.00095), o melhor resultado publicado no essay-br: BERTimbau base,
tema e redacao na entrada (separados pelo [SEP]), regressao, ate 5 epocas, lote 16. A epoca
final e a de maior QWK total na validacao. O BERT le no maximo 512 tokens: o fim das redacoes
longas e cortado (limite conhecido; afeta a C5, que fica no fim).

Saida no formato do run_api_scoring.py, para o evaluate.py e o calibrate.py:
{out}_val.csv e {out}_test.csv.

    python train_bertimbau.py --train data/cp_train.csv --val data/cp_val.csv --test data/cp_test.csv \\
        --out results/colab/bertimbau_cp
"""
from __future__ import annotations

import argparse
import math
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from evaluate import evaluate_pair
from run_api_scoring import limpar

BASE = "neuralmind/bert-base-portuguese-cased"
NOTAS = ["c1", "c2", "c3", "c4", "c5"]


def tabela(df, pred, modelo):
    """Predicoes (n, 5) em 0 a 1 -> CSV no formato do run_api_scoring.py."""
    p = np.clip(pred, 0, 1) * 200
    out = df[["index_redacao", "score"] + NOTAS].copy()
    for j, c in enumerate(NOTAS):
        out[f"pred_{c}"] = p[:, j].round(1)
    out["pred_total"] = p.sum(1).round(1)
    out["ok"], out["modelo"], out["modo"], out["raw"] = 1, modelo, "regressao", ""
    out["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True, help="prefixo dos CSVs de saida")
    ap.add_argument("--temas", default="data/prompts_essaybr.csv")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="usa so N redacoes de cada parte (teste)")
    args = ap.parse_args(argv)

    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments)

    temas = pd.read_csv(args.temas).set_index("id")["title"]
    tok = AutoTokenizer.from_pretrained(args.base)
    partes = {n: pd.read_csv(p) for n, p in (("train", args.train), ("val", args.val), ("test", args.test))}
    if args.limit:
        partes = {n: d.head(args.limit) for n, d in partes.items()}

    def dataset(df):
        enc = tok([str(temas.get(t, "")) for t in df["prompt"]], [limpar(e) for e in df["essay"]],
                  truncation="only_second", max_length=args.max_length)
        enc["labels"] = (df[NOTAS].to_numpy(float) / 200).tolist()
        return Dataset.from_dict(dict(enc))

    ds = {n: dataset(d) for n, d in partes.items()}

    def qwk_total(ev):
        pred, gold = np.asarray(ev.predictions), np.asarray(ev.label_ids)
        return {"qwk": evaluate_pair(gold.sum(1) * 200, np.clip(pred, 0, 1).sum(1) * 200, is_total=True)["qwk"]}

    passos = math.ceil(len(ds["train"]) / 16) * args.epochs
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=5, problem_type="regression")
    trainer = Trainer(
        model=model, processing_class=tok, data_collator=DataCollatorWithPadding(tok),
        train_dataset=ds["train"], eval_dataset=ds["val"], compute_metrics=qwk_total,
        args=TrainingArguments(
            output_dir=tempfile.mkdtemp(), num_train_epochs=args.epochs, learning_rate=args.lr,
            per_device_train_batch_size=16, per_device_eval_batch_size=32, weight_decay=0.01,
            warmup_steps=max(1, round(0.1 * passos)), eval_strategy="epoch", save_strategy="epoch",
            load_best_model_at_end=True, metric_for_best_model="qwk", greater_is_better=True,
            save_total_limit=1, logging_steps=50, report_to="none", bf16=torch.cuda.is_available()))
    trainer.train()
    hist = [(round(h["epoch"]), round(h["eval_qwk"], 3)) for h in trainer.state.log_history if "eval_qwk" in h]
    print(f"QWK na validacao por epoca: {hist} | melhor: {max(hist, key=lambda x: x[1])}")

    nome = args.base.rstrip("/").split("/")[-1]
    for n in ("val", "test"):
        pred = trainer.predict(ds[n]).predictions
        tabela(partes[n], pred, nome).to_csv(f"{args.out}_{n}.csv", index=False)
    print(f"feito: {args.out}_val.csv e {args.out}_test.csv")


if __name__ == "__main__":
    main()
