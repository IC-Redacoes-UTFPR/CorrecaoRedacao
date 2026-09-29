"""Features do LanguageTool (offline, pt-BR) para a competencia 1.

Base: "Improve LLM-based AES with Linguistic Features" (arXiv 2502.09497). O corretor conta os
desvios da redacao e o resultado entra como contexto no prompt de C1 (modo mts_lt do
run_api_scoring.py). Roda local, nao gasta cota de API.

Precisa de Java 17+. Na primeira vez o pacote baixa o LanguageTool (~260 MB).

Saida: CSV com index_redacao, palavras, desvios, desvios_100 (por 100 palavras), contagem por
categoria e ate 8 exemplos (trecho + mensagem) em JSON.

    pip install language_tool_python
    python lt_features.py                                     # data/amostra_300.csv
    python lt_features.py --amostra data/val_c5.csv --out data/lt_val.csv
"""
from __future__ import annotations

import argparse
import ast
import json

import pandas as pd

from run_api_scoring import limpar

# So categorias de norma culta. Ficam de fora TYPOGRAPHY (espaco antes de pontuacao e
# artefato do essay-br, "pais ."), REPETITIONS ("anos anos" sobra das correcoes do corpus)
# e estilo (STYLE, REDUNDANCY, SHORTEN_IT, FORMAL, OBJECTIVE), que e gosto, nao desvio.
CATEGORIAS = {"TYPOS": "ortografia", "MISSPELLING": "ortografia", "GRAMMAR": "gramatica",
              "SYNTAX": "gramatica", "PUNCTUATION": "pontuacao", "CASING": "maiusculas",
              "CONFUSED_WORDS": "palavras_confundidas"}
IGNORAR = {"UPPERCASE_SENTENCE_START"}  # paragrafos do corpus quebram no meio da frase


def conta(m, texto):
    if m.category not in CATEGORIAS or m.rule_id in IGNORAR:
        return False
    # ortografia so em palavra minuscula: sigla e nome proprio (DUDH, Thrax, covid) nao e erro
    trecho = texto[m.offset: m.offset + m.error_length]
    return m.category != "TYPOS" or trecho[:1].islower() and trecho.lower() != "covid"


def texto_lt(essay):
    """Junta a lista de linhas do essay-br com espaco. O limpar() deixa os separadores
    "', '" no meio do texto, que o LanguageTool conta como erro de pontuacao."""
    try:
        linhas = ast.literal_eval(essay)
    except (ValueError, SyntaxError):
        return limpar(essay)
    return limpar(" ".join(str(x).strip() for x in linhas if str(x).strip()))


def extrair(tool, texto):
    ms = [m for m in tool.check(texto) if conta(m, texto)]
    palavras = max(len(texto.split()), 1)
    cont = {v: 0 for v in CATEGORIAS.values()}
    for m in ms:
        cont[CATEGORIAS[m.category]] += 1
    exemplos = [{"trecho": texto[m.offset: m.offset + m.error_length], "msg": m.message}
                for m in ms[:8]]
    return {"palavras": palavras, "desvios": len(ms),
            "desvios_100": round(100 * len(ms) / palavras, 2), **cont,
            "exemplos": json.dumps(exemplos, ensure_ascii=False)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amostra", default="data/amostra_300.csv")
    ap.add_argument("--out", default="data/lt_amostra_300.csv")
    args = ap.parse_args(argv)

    import language_tool_python
    tool = language_tool_python.LanguageTool("pt-BR")
    df = pd.read_csv(args.amostra)
    linhas = []
    for i, row in enumerate(df.itertuples()):
        linhas.append({"index_redacao": row.index_redacao, **extrair(tool, texto_lt(row.essay))})
        if i % 50 == 0:
            print(f"{i}/{len(df)}")
    tool.close()
    out = pd.DataFrame(linhas)
    out.to_csv(args.out, index=False)
    print(f"{len(out)} redacoes -> {args.out}")

    # sinal bruto: o quanto a contagem acompanha a nota humana de C1 (negativo e bom)
    g = df.set_index("index_redacao")["c1"].reindex(out["index_redacao"]).to_numpy()
    for col in ["desvios", "desvios_100", *dict.fromkeys(CATEGORIAS.values())]:
        x = out[col] if col.startswith("desvios") else 100 * out[col] / out["palavras"]
        rho = pd.Series(x.to_numpy()).corr(pd.Series(g), method="spearman")
        print(f"spearman(c1, {col}) = {rho:+.3f}")


if __name__ == "__main__":
    main()
