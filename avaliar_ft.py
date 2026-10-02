"""Avalia o gpt-oss-20b com fine-tuning (adaptador LoRA do finetune_gptoss.py) direto no
transformers, em lote, sem vLLM.

Carrega o modelo base, aplica o adaptador salvo em results/colab/ft_lora (vem do GitHub, entao
funciona numa maquina nova sem treinar de novo), mescla e gera as 5 notas com o mesmo
prompt_ft do treino. Saida no formato do run_api_scoring.py, para o evaluate.py e o calibrate.py.

    python avaliar_ft.py --amostra data/cp_test.csv --out results/colab/gptoss20b_ft_cp.csv
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import pandas as pd

import run_api_scoring as ras
from run_api_scoring import COMPS, limpar, parse_notas

BASE = "openai/gpt-oss-20b"


def notas_esperadas(scores, tokens, j, ids, bandas):
    """Para cada passo em que a linha j gerou uma nota, a media das notas ponderada pela
    probabilidade (softmax dos logits so entre os tokens das 6 notas)."""
    import torch
    return [float((torch.softmax(scores[s][j, ids].float(), -1).cpu() * bandas).sum())
            for s, t in enumerate(tokens) if t in set(ids)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amostra", default="data/cp_test.csv")
    ap.add_argument("--adaptador", default="results/colab/ft_lora")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--temas", default="data/prompts_essaybr.csv")
    ap.add_argument("--out", default="results/colab/gptoss20b_ft_cp.csv")
    ap.add_argument("--lote", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--esperado", action="store_true",
                    help="nota = media ponderada pela probabilidade das 6 notas, nao so a mais provavel")
    args = ap.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

    t = pd.read_csv(args.temas)
    ras.TEMAS = {int(r.id): (r.title, str(r.description)) for r in t.itertuples()}
    df = pd.read_csv(args.amostra)
    df = df.head(args.limit) if args.limit else df
    feitos = set(pd.read_csv(args.out)["index_redacao"]) if os.path.exists(args.out) else set()
    df = df[~df["index_redacao"].isin(feitos)]
    print(f"{len(df)} redacoes a avaliar ({len(feitos)} ja feitas)")

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "left"  # geracao em lote precisa do padding a esquerda
    cfg = AutoConfig.from_pretrained(args.base)
    quant = Mxfp4Config(dequantize=True) if getattr(cfg, "quantization_config", None) else None
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, quantization_config=quant,
                                                 device_map="auto")
    model = PeftModel.from_pretrained(model, args.adaptador).merge_and_unload().eval()

    # cada nota (0, 40, ..., 200) e um token so no tokenizer do gpt-oss; e o que permite ler a
    # probabilidade de cada nota numa posicao
    bandas = torch.tensor([0., 40., 80., 120., 160., 200.])
    ids = [tok.encode(str(int(b)), add_special_tokens=False) for b in bandas]
    assert all(len(i) == 1 for i in ids), f"nota com mais de um token: {ids}"
    ids = [i[0] for i in ids]

    novo = not os.path.exists(args.out)
    cols = ["index_redacao", "score", "c1", "c2", "c3", "c4", "c5", "pred_c1", "pred_c2", "pred_c3", "pred_c4",
            "pred_c5", "pred_total", "ok", "modelo", "modo", "ts", "raw"]
    for i in range(0, len(df), args.lote):
        lote = df.iloc[i:i + args.lote]
        textos = [tok.apply_chat_template([{"role": "user", "content": ras.prompt_ft(limpar(r.essay), r.prompt)}],
                                          tokenize=False, add_generation_prompt=True) for r in lote.itertuples()]
        enc = tok(textos, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id,
                                 return_dict_in_generate=True, output_scores=args.esperado)
        gerado = out.sequences[:, enc["input_ids"].shape[1]:]
        respostas = tok.batch_decode(gerado, skip_special_tokens=True)
        linhas = []
        for j, (r, resp) in enumerate(zip(lote.itertuples(), respostas)):
            n = parse_notas(resp)
            if n and args.esperado:
                # nas posicoes em que o modelo escreveu uma nota, troca a nota mais provavel pela
                # media das 6 notas ponderada pela probabilidade de cada uma (o modelo treinado
                # quase so escolhe as 2 notas do meio; a probabilidade guarda a nuance)
                ev = notas_esperadas(out.scores, gerado[j].tolist(), j, ids, bandas)
                if len(ev) >= len(COMPS):
                    n = {c: round(v, 1) for c, v in zip(COMPS, ev)}
                    n["Nota_Total"] = round(sum(n.values()), 1)
            linhas.append({"index_redacao": r.index_redacao, "score": r.score,
                           **{c: getattr(r, c) for c in ["c1", "c2", "c3", "c4", "c5"]},
                           **{f"pred_{c.lower()}": (n[c] if n else "") for c in COMPS},
                           "pred_total": n["Nota_Total"] if n else "", "ok": int(bool(n)),
                           "modelo": "gpt-oss-20b-ft-cp", "modo": "ft_esperado" if args.esperado else "ft",
                           "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           "raw": "" if n else resp[:500]})
        pd.DataFrame(linhas, columns=cols).to_csv(args.out, mode="a", header=novo, index=False)
        novo = False
        print(f"{min(i + args.lote, len(df))}/{len(df)} | ok {sum(l['ok'] for l in linhas)}/{len(linhas)} "
              f"| exemplo: {respostas[0][:80]!r}")
    print(f"feito. saida em {args.out}")


if __name__ == "__main__":
    main()
