"""Fine-tuning (LoRA) do gpt-oss-20b para dar as 5 notas, no treino cross-prompt.

Receita do cookbook da OpenAI ("Fine-tuning with Hugging Face Transformers"): pesos MXFP4
desquantizados para bf16, LoRA r=8 em todas as camadas lineares e nos experts de 3 camadas,
TRL SFTTrainer. Cabe numa H100 80 GB (ou G4 96 GB).

Entrada: data/cp_train.csv (temas disjuntos do teste, ver make_cross_prompt_split.py).
Cada exemplo: prompt = prompt_ft(redacao, tema) do run_api_scoring.py (o mesmo usado na
avaliacao), resposta = JSON com as 5 notas humanas. A perda e so na resposta.

Saida: o adaptador LoRA final em --out e o de cada epoca em --out_ep1, --out_ep2, ..., para
escolher a epoca pela validacao (avaliar_ft.py), como em Barbosa et al. (PROPOR 2026), que
treinam ate 20 epocas e ficam com a melhor no QWK de validacao.

    python finetune_gptoss.py --out /content/ft6_lora --epochs 6
    python finetune_gptoss.py --dry-run     # so monta e mostra 1 exemplo, sem GPU
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import pandas as pd

import run_api_scoring as ras

BASE = "openai/gpt-oss-20b"


def exemplos(train_csv, temas_csv):
    t = pd.read_csv(temas_csv)
    ras.TEMAS = {int(r.id): (r.title, str(r.description)) for r in t.itertuples()}
    df = pd.read_csv(train_csv)
    return [{"prompt": [{"role": "user", "content": ras.prompt_ft(ras.limpar(r.essay), r.prompt)}],
             "completion": [{"role": "assistant", "content": json.dumps(
                 {c: int(getattr(r, c.lower())) for c in ras.COMPS})}]}
            for r in df.itertuples()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="data/cp_train.csv")
    ap.add_argument("--temas", default="data/prompts_essaybr.csv")
    ap.add_argument("--out", default="/content/ft_lora")
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--base", default=BASE, help="modelo base (um gpt-oss minusculo local serve de teste)")
    ap.add_argument("--limit", type=int, default=0, help="usa so N exemplos (teste)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    dados = exemplos(args.train, args.temas)
    dados = dados[:args.limit] if args.limit else dados
    print(f"{len(dados)} exemplos de treino")
    if args.dry_run:
        print(json.dumps(dados[0], ensure_ascii=False)[:1500])
        return

    import math

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(args.base)
    cfg = AutoConfig.from_pretrained(args.base)
    # o checkpoint oficial e MXFP4: desquantiza para bf16 para treinar (sem isso nao ha gradiente)
    quant = Mxfp4Config(dequantize=True) if getattr(cfg, "quantization_config", None) else None
    model = AutoModelForCausalLM.from_pretrained(
        args.base, attn_implementation="eager", dtype=torch.bfloat16,
        quantization_config=quant, use_cache=False, device_map="auto")
    # experts de 1 a cada 8 camadas, como no cookbook (7, 15, 23 no modelo de 24 camadas)
    camadas = [i for i in range(cfg.num_hidden_layers) if i % 8 == 7]
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, target_modules="all-linear",
        target_parameters=[f"{i}.mlp.experts.{p}" for i in camadas for p in ("gate_up_proj", "down_proj")]))
    model.print_trainable_parameters()

    passos = math.ceil(len(dados) / 16 * args.epochs)  # 16 = batch 4 x acumulacao 4
    trainer = SFTTrainer(
        model=model, processing_class=tok, train_dataset=Dataset.from_list(dados),
        args=SFTConfig(
            output_dir=args.out, num_train_epochs=args.epochs, learning_rate=2e-4,
            per_device_train_batch_size=4, gradient_accumulation_steps=4, gradient_checkpointing=True,
            max_length=2048, warmup_steps=max(1, round(0.03 * passos)),  # warmup_ratio saiu do transformers 5
            lr_scheduler_type="cosine_with_min_lr", lr_scheduler_kwargs={"min_lr_rate": 0.1},
            logging_steps=10, save_strategy="epoch", save_only_model=True, report_to="none",
            bf16=torch.cuda.is_available()))
    trainer.train()
    trainer.save_model(args.out)

    # um adaptador por epoca (o Trainer salva checkpoint-<passo> ao fim de cada uma)
    cks = sorted(glob.glob(f"{args.out}/checkpoint-*"), key=lambda d: int(d.rsplit("-", 1)[1]))
    for ep, ck in enumerate(cks, 1):
        os.makedirs(f"{args.out}_ep{ep}", exist_ok=True)
        for f in ("adapter_config.json", "adapter_model.safetensors"):
            shutil.copy(f"{ck}/{f}", f"{args.out}_ep{ep}/{f}")
    print(f"adaptador final em {args.out}; por epoca em {args.out}_ep1 a _ep{len(cks)}")


if __name__ == "__main__":
    main()
