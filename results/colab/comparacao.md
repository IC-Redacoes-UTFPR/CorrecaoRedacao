# Comparacao no essay-br estendido

QWK do total no conjunto de teste; por competencia, QWK bruto.

## Nossos modelos

| Divisao | Modelo | QWK bruto | Calibrado pela val. (desloc.) | Calibrado pela val. (quantil) | Pearson | C1 | C2 | C3 | C4 | C5 |
|---|---|---|---|---|---|---|---|---|---|---|
| oficial | BERTimbau base (regressao) | 0,74 | 0,77 | 0,77 | 0,78 | 0,58 | 0,62 | 0,62 | 0,67 | 0,68 |
| oficial | gpt-oss-20b LoRA, nota esperada | 0,78 | 0,78 | 0,78 | 0,79 | 0,64 | 0,61 | 0,62 | 0,71 | 0,73 |
| cross-prompt | gpt-oss-120b sem treino (mts_fs) | 0,45 | 0,62 | 0,60 | 0,63 | 0,21 | 0,46 | 0,47 | 0,31 | 0,25 |
| cross-prompt | BERTimbau base (regressao) | 0,68 | 0,68 | 0,71 | 0,72 | 0,42 | 0,51 | 0,46 | 0,58 | 0,56 |
| cross-prompt | gpt-oss-20b LoRA 3 epocas, nota esperada | 0,66 | 0,67 | 0,69 | 0,69 | 0,49 | 0,53 | 0,49 | 0,54 | 0,47 |
| cross-prompt | gpt-oss-20b LoRA ft6 (epoca pela validacao) | 0,66 | 0,67 | 0,67 | 0,68 | 0,50 | 0,55 | 0,49 | 0,56 | 0,48 |

## Literatura (mesmo corpus, divisao aleatoria: temas do teste vistos no treino)

| Divisao | Modelo | QWK bruto | Calibrado pela val. (desloc.) | Calibrado pela val. (quantil) | Pearson | C1 | C2 | C3 | C4 | C5 |
|---|---|---|---|---|---|---|---|---|---|---|
| oficial | Amorim e Veloso (2017), features (Marinho et al., JIDM 2022) | 0,49 | | | | 0,39 | 0,46 | 0,40 | 0,38 | 0,34 |
| oficial | Fonseca et al. (2018), features (Marinho et al., JIDM 2022) | 0,53 | | | | 0,44 | 0,48 | 0,42 | 0,47 | 0,38 |
| aleatoria 70/15/15, semente nao informada | BERTimbau base (Matsuoka, arXiv 2023, sem revisao) | 0,79 | | | | 0,74 | 0,78 | 0,76 | 0,84 | 0,79 |
