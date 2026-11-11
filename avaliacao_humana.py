"""Kit da avaliacao humana do feedback e calculo da concordancia.

Montar o kit (60 feedbacks v3 com a nota do ft3, que e o sistema real, em ordem embaralhada):

    python avaliacao_humana.py --montar

Gera em avaliacao_humana/:
- itens.html: cada feedback como o aluno veria (problemas agrupados por trecho), com codigo F01..F60.
  Abre no navegador; da para imprimir em PDF.
- planilha_avaliador.csv: uma linha por feedback, 4 criterios de 1 a 5 e comentario. Uma copia por
  avaliador (separador ";" para abrir direto no Excel em portugues).
- instrucoes.md: a rubrica.
- chave.csv: codigo -> redacao e arquivo de origem. NAO entregar aos avaliadores.

Depois de preenchidas, concordancia entre dois avaliadores e com o juiz automatico:

    python avaliacao_humana.py --concordancia avaliador_A.csv avaliador_B.csv
"""
from __future__ import annotations

import argparse
import html
import json
import os

import numpy as np
import pandas as pd

from evaluate import _spearman, qwk
from gerar_feedback import por_trecho
from run_api_scoring import COMPS, paragrafos

OUT = "avaliacao_humana"
FONTES = [("results/colab/feedback_cp30_comp3_ft3.jsonl", "data/amostra_feedback_cp30.csv",
           "results/juiz/cp30_v3_ft3.jsonl"),
          ("results/colab/feedback_cp30b_comp3_ft3.jsonl", "data/amostra_feedback_cp30b.csv",
           "results/juiz/cp30b_v3_ft3.jsonl")]
CRITERIOS = ["especificidade", "acionabilidade", "correcao", "alinhamento"]
SEMENTE = 2026
# RUBRICA do run_api_scoring.py e sem acento (texto de prompt); aqui e para leitores de Letras
LEGENDA = {
    "C1": "Domínio da modalidade escrita formal da língua portuguesa (ortografia, concordância, "
          "regência, pontuação, escolha de registro).",
    "C2": "Compreensão da proposta e aplicação do tipo textual dissertativo-argumentativo para "
          "desenvolver o tema, sem tangenciar nem fugir.",
    "C3": "Seleção, relação, organização e interpretação de informações, fatos, opiniões e argumentos "
          "em defesa de um ponto de vista.",
    "C4": "Conhecimento dos mecanismos linguísticos de coesão: articulação entre parágrafos, períodos e "
          "partes do texto (conectivos, referenciação).",
    "C5": "Elaboração de proposta de intervenção para o problema, detalhada e respeitando os direitos "
          "humanos, com agente, ação, modo ou meio, efeito e detalhamento.",
}

INSTRUCOES = """# Avaliação do feedback: instruções

Você vai ler 60 feedbacks escritos por um sistema automático para redações do ENEM. Cada um tem um
código (F01 a F60), a redação do estudante, a nota por competência que o sistema deu e o feedback.
Avalie **o feedback**, não a redação. Preencha a planilha com uma nota de 1 a 5 em cada critério.
Não converse com o outro avaliador sobre os itens até os dois terminarem.

| Critério | 1 | 3 | 5 |
|---|---|---|---|
| **Especificidade** | genérico, serviria para qualquer redação | aponta alguns trechos, mas mistura com comentários vagos | aponta trechos e problemas concretos desta redação |
| **Acionabilidade** | o estudante não saberia o que fazer | algumas orientações claras, outras vagas | orientações claras que o estudante consegue aplicar na próxima redação |
| **Correção** | afirmações falsas sobre a redação, problemas que não existem ou na competência errada | alguns erros do avaliador | tudo o que o feedback diz é verdade e está na competência certa |
| **Alinhamento com a nota** | o feedback contradiz a nota (ex.: nota alta com muitos problemas graves) | parcialmente coerente | a gravidade dos problemas e dos elogios acompanha a nota de cada competência |

Use 2 e 4 para os casos intermediários. O campo "comentario" é opcional: use para apontar um erro
grave do feedback ou algo que chamou atenção.

Tempo estimado: 5 a 8 minutos por feedback.
"""

CSS = ("body{font-family:Georgia,serif;max-width:820px;margin:auto;padding:16px;line-height:1.5;"
       "color:#222;background:#fff}h2{border-top:3px solid #333;padding-top:12px;page-break-before:always}"
       "blockquote{background:#f4f4f4;padding:8px 14px;margin:0}table{border-collapse:collapse}"
       "td,th{border:1px solid #999;padding:3px 8px}.c{font-weight:bold;color:#444}")


def esc(s):
    return html.escape(str(s or ""))


def item_html(cod, r, a):
    notas = r["notas"]
    fb = r["feedback"]
    essay = paragrafos(a["essay"])
    h = [f"<h2>{cod}</h2>", f"<p><b>Tema:</b> {esc(a['title'])}</p>",
         "<table><tr>" + "".join(f"<th>{c}</th>" for c in COMPS) + "<th>Total</th></tr><tr>"
         + "".join(f"<td>{notas[c]}</td>" for c in COMPS) + f"<td>{sum(notas.values())}</td></tr></table>",
         "<h3>Redação</h3><blockquote>" + "".join(f"<p>{esc(p)}</p>" for p in essay.split("\n\n")) + "</blockquote>",
         "<h3>Pontos fortes</h3><ul>"]
    h += [f"<li><span class=c>{c}:</span> {esc(p)}</li>" for c in COMPS for p in (fb.get(c) or {}).get("pontos_fortes") or []]
    h.append("</ul><h3>O que corrigir, trecho a trecho</h3><ol>")
    for g in por_trecho(fb, essay):
        h.append(f"<li>&ldquo;{esc(g['trecho'])}&rdquo;<ul>")
        for c, expl, cor in g["itens"]:
            h.append(f"<li><span class=c>{c}:</span> {esc(expl)}"
                     + (f"<br><i>Sugestão:</i> {esc(cor)}" if cor else "") + "</li>")
        h.append("</ul></li>")
    h.append("</ol><h3>Como melhorar</h3><ul>")
    h += [f"<li><span class=c>{c}:</span> {esc(p)}</li>" for c in COMPS for p in (fb.get(c) or {}).get("como_melhorar") or []]
    h.append("</ul>")
    return "\n".join(h)


def montar():
    itens = []
    for fb_path, am_path, _ in FONTES:
        am = pd.read_csv(am_path).set_index("index_redacao")
        for l in open(fb_path, encoding="utf-8"):
            r = json.loads(l)
            itens.append((r, am.loc[r["index_redacao"]], fb_path))
    ordem = np.random.default_rng(SEMENTE).permutation(len(itens))
    os.makedirs(OUT, exist_ok=True)
    corpo, chave = [], []
    for k, i in enumerate(ordem):
        r, a, fonte = itens[i]
        cod = f"F{k + 1:02d}"
        corpo.append(item_html(cod, r, a))
        chave.append({"codigo": cod, "index_redacao": r["index_redacao"], "feedback": fonte})
    leg = "".join(f"<li><b>{c}</b>: {esc(LEGENDA[c])}</li>" for c in COMPS)
    open(f"{OUT}/itens.html", "w", encoding="utf-8").write(
        f"<!doctype html><html lang=pt-BR><meta charset=utf-8><title>Feedbacks para avaliar</title>"
        f"<style>{CSS}</style><body><h1>Feedbacks para avaliar</h1><p>Competências:</p><ul>{leg}</ul>"
        + "\n".join(corpo) + "</body></html>")
    pd.DataFrame(chave).to_csv(f"{OUT}/chave.csv", index=False)
    pd.DataFrame({"codigo": [c["codigo"] for c in chave], **{k: "" for k in CRITERIOS}, "comentario": ""}) \
        .to_csv(f"{OUT}/planilha_avaliador.csv", sep=";", index=False, encoding="utf-8-sig")
    open(f"{OUT}/instrucoes.md", "w", encoding="utf-8").write(INSTRUCOES)
    print(f"{len(chave)} feedbacks -> {OUT}/ (itens.html, planilha_avaliador.csv, instrucoes.md, chave.csv)")


def ler(p):
    df = pd.read_csv(p, sep=";", encoding="utf-8-sig").set_index("codigo")
    return df[CRITERIOS].apply(pd.to_numeric, errors="coerce")


def juiz_por_codigo():
    """Media do juiz nas 5 competencias, nos criterios que ele tem (sem alinhamento)."""
    chave = pd.read_csv(f"{OUT}/chave.csv")
    j = {}
    for fb_path, _, juiz_path in FONTES:
        for l in open(juiz_path, encoding="utf-8"):
            r = json.loads(l)
            n = r["notas"]
            j[(fb_path, r["index_redacao"])] = {
                k: np.mean([n[c][jk] for c in COMPS]) for k, jk in
                [("especificidade", "especificidade"), ("acionabilidade", "utilidade"), ("correcao", "validade")]}
    return pd.DataFrame({row.codigo: j.get((row.feedback, row.index_redacao), {}) for row in chave.itertuples()}).T


def concordancia(a_path, b_path):
    a, b = ler(a_path), ler(b_path)
    juiz = juiz_por_codigo()
    print("| Criterio | n | QWK entre avaliadores | Acordo exato | Acordo adjacente (+-1) | Media A | Media B | Spearman humanos x juiz |")
    print("|---|---|---|---|---|---|---|---|")
    for k in CRITERIOS:
        ok = a[k].notna() & b[k].reindex(a.index).notna()
        x, y = a.loc[ok, k].astype(int), b[k].reindex(a.index)[ok].astype(int)
        q = qwk(x - 1, y - 1, 5)
        sj = ""
        if k in juiz.columns:
            m = ((x + y) / 2)
            sj = f"{_spearman(m.to_numpy(), juiz.loc[m.index, k].to_numpy(float)):.2f}"
        print(f"| {k} | {ok.sum()} | {q:.2f} | {100 * (x == y).mean():.0f}% | {100 * ((x - y).abs() <= 1).mean():.0f}% | "
              f"{x.mean():.2f} | {y.mean():.2f} | {sj} |")


def autoteste():
    """Planilhas simuladas: copia identica da QWK 1; o juiz tem que casar com todos os codigos."""
    import tempfile
    p = pd.read_csv(f"{OUT}/planilha_avaliador.csv", sep=";", encoding="utf-8-sig")
    rng = np.random.default_rng(0)
    for k in CRITERIOS:
        p[k] = rng.integers(1, 6, len(p))
    d = tempfile.mkdtemp()
    p.to_csv(f"{d}/a.csv", sep=";", index=False, encoding="utf-8-sig")
    assert (ler(f"{d}/a.csv") == p.set_index("codigo")[CRITERIOS]).all().all()
    assert qwk(p.especificidade - 1, p.especificidade - 1, 5) == 1.0
    assert juiz_por_codigo().notna().all().all() and len(juiz_por_codigo()) == len(p)
    concordancia(f"{d}/a.csv", f"{d}/a.csv")
    print("autoteste ok")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--montar", action="store_true")
    ap.add_argument("--concordancia", nargs=2, metavar=("A", "B"))
    ap.add_argument("--autoteste", action="store_true")
    args = ap.parse_args(argv)
    if args.montar:
        montar()
    if args.concordancia:
        concordancia(*args.concordancia)
    if args.autoteste:
        autoteste()


if __name__ == "__main__":
    main()
