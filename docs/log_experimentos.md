# Log de experimentos, reteste com APIs gratuitas

Registro datado de cada teste e do estado das cotas diárias das APIs gratuitas.
Objetivo: saber a qualquer momento o que já foi rodado, com que resultado, e quanta
cota resta no dia.

Todos os QWK são na nota total, avaliação cross-prompt (folds por tema), 300 redações
do conjunto de teste do v5 salvo em `data/amostra_300.csv`, salvo indicação de `n`
menor. "Calibrado" = deslocamento de viés aprendido out-of-fold (`calibrate.py`).

## Resultados por data

| Data | Modelo | Modo | n | QWK bruto | QWK calibrado | Pearson | Observação |
|---|---|---|---|---|---|---|---|
| 2026-09-02 | Qwen 2.5 7B | zero-shot | 1293 | 0,35 | 0,35 | 0,37 | melhor resultado anterior do projeto, recalculado |
| 2026-09-02 | Gemma 2 9B | few-shot | 1297 | 0,25 | 0,42 | 0,44 | melhor 7B após calibração. Achado: erro é viés de escala, não discriminação |
| 2026-09-03 | Gemini 3.5 Flash Lite | holístico | 300 | 0,47 | 0,54 | 0,55 | primeiro resultado por API a bater a linha de base |
| 2026-09-03 | Gemini 3.5 Flash Lite | MTS | 298 | 0,53 | 0,60 | 0,60 | melhor resultado. 0,60 é o piso da literatura para essay-br |
| 2026-09-03 | gpt-oss-120B (Groq) | holístico | 300 | 0,23 | 0,50 | 0,55 | viés bruto -226. Discrimina igual ao Flash Lite, escala muito pior |
| 2026-09-03 | Gemini 3.8 Flash | holístico | 60 | 0,40 | 0,42 | 0,42 | modelo preview, RPD real ~13 por conta. Pior que o Flash Lite. Descartado |
| 2026-09-04 | Gemini 3.5 Flash Lite | MTS2 (C1/C5 dedicado) | 257 | 0,54 | 0,60 | 0,60 | prompt estruturado para C1 e C5. Não melhorou. C1 piorou (0,29 para 0,21), C5 igual. Descartado |
| 2026-09-04 | verificação de métricas | - | - | - | - | - | `verify_metrics.py`: QWK, Pearson, Spearman e MAE batem com scikit-learn/scipy e com o valor calculado pelo notebook v5 |
| 2026-09-08 | Gemini 3.5 Flash Lite | MTS_FS (few-shot com âncoras) | 297 | 0,58 | 0,60 | 0,62 | uma redação âncora por faixa de nota por competência. Sobe a discriminação bruta (Pearson 0,60 para 0,62) e reduz o viés (-82 para -59), mas o total calibrado empata em 0,60. C1 0,29 para 0,34, C3 e C4 sobem, C5 travado em 0,33 |
| 2026-09-09 | Gemini 3.5 Flash Lite | MTS_FS2 (âncoras + checklist no C5) | 296 | 0,59 | 0,61 | 0,60 | âncoras em C1 a C4, mais checklist dos 5 elementos e aviso "parcial não é 0" só no C5. C5 sai de 0,33 para 0,35 e o viés do C5 quase some (média 90 para 113, humano 119). Melhor config até agora: QWK bruto 0,59, viés geral -36. Total calibrado 0,605. Adotado como padrão |
| 2026-09-29 | Gemini 3.5 Flash Lite | MTS_RR (Reflect-and-Revise no C5) | 292 | 0,62 | 0,60 | 0,62 | rubrica de C5 reescrita pelo próprio modelo em 5 iterações num fold de validação. C5 0,35 para 0,39 (IC95 da diferença [-0,01, +0,08], não significativo). O ganho do bruto vem do viés: C5 virou generoso (+12) e compensa o viés negativo de C1 e C4; calibrado empata. Rodado pela colega |
| 2026-09-29 | Gemini 3.5 Flash Lite | MTS_LT (LanguageTool no C1) | 293 | 0,60 | 0,61 | 0,61 | contagem de desvios do LanguageTool e exemplos no prompt de C1. C1 0,35 para 0,38 (IC95 [-0,01, +0,09], não significativo), viés de C1 -21 para -16. Total praticamente igual |
| 2026-09-29 | Gemini 3.5 Flash Lite | MTS_LT_RR (C1 do LT + C5 da v5) | 289 | 0,63 | 0,61 | 0,63 | melhor bruto e melhor Pearson até agora. QWK bruto +0,03 sobre o MTS_FS2 (IC95 [+0,01, +0,06], significativo), mas quase todo por viés; Pearson +0,02 (IC95 [0,00, +0,04]) no limite. Calibrado 0,612 contra 0,605: empate |
| 2026-09-30 | gpt-oss-20B (vLLM, Colab A100) | MTS_LT | 286 | 0,27 | 0,39 | 0,45 | primeiro teste local no Colab. Bem pior que o Flash Lite. C1 desmonta (QWK 0,01, média 49 contra 136 humano: o modelo fica severo demais com a contagem do LanguageTool), C5 generoso demais (+66). 14 redações sem nota (falha de parse). Tempo não medido: a rodada foi interrompida e retomada. Descartado; próximo é o gpt-oss-120B em H100 |
| 2026-09-30 | gpt-oss-120B (vLLM, Colab G4 / RTX PRO 6000 96 GB) | MTS_LT | 300 | 0,58 | 0,56 | 0,58 | reasoning low, 12 processos, 300 redações em 6 min (~1 unidade de computação), nenhuma falha de parse. Perto do Flash Lite no total, mas por compensação: C1 severo (0,16, viés -47) e C5 generoso (0,19, viés +52) se anulam. C2 0,49, C3 0,42, C4 0,42, iguais ou melhores que o Flash Lite. O calibrado cai porque o viés total já é pequeno (-14) e a correção por tema só adiciona ruído. Na G4 (Blackwell) o vLLM precisa de `--attention-backend TRITON_ATTN` e `VLLM_USE_FLASHINFER_SAMPLER=0` |
| 2026-09-30 | gpt-oss-120B (vLLM, Colab G4) | MTS_LT, reasoning medium | 298 | 0,56 | 0,58 | 0,59 | `--reasoning medium --max-tokens 4000`, 23 min (4x o low). Não ajuda: Pearson 0,58 para 0,59, dentro do ruído. C1 piora (0,16 para 0,07, viés -47 para -70): pensar mais deixa o modelo ainda mais severo com a lista do LanguageTool. C5 igual (0,19, viés +58). Fica o low |
| 2026-10-01 | gpt-oss-120B (vLLM, Colab G4) | MTS_FS | 300 | 0,52 | 0,60 | 0,60 | âncoras nas 5 competências, sem LanguageTool e sem checklist. Empata com o Flash Lite (0,605 calibrado). Tirar o checklist do C5 é o que mais ajuda: C5 0,19 para 0,39, viés +52 para -27. C1 segue fraco (0,20 contra 0,35 do Flash Lite) e o viés total volta a ser negativo (-91), que a calibração corrige. A primeira rodada misturou 103 linhas do 20b (MODEL errado no notebook); foram removidas e refeitas, e o script agora recusa retomar CSV de outro modelo |
| 2026-10-01 | gpt-oss-120B (vLLM, Colab G4) | MTS_FS com 3 âncoras por faixa | 299 | 0,45 | 0,59 | 0,60 | `--anchors data/anchors_k3.csv` (90 âncoras, inclui as 30 do k=1), 5 min. Empata com o k=1 (0,599): C1 sobe (0,20 para 0,26), mas C2 (0,44 para 0,39) e C5 (0,39 para 0,32) caem, e o viés piora (-91 para -127). Usar o C1 do k=3 com o resto do k=1 dá 0,596: sem ganho. Fica o k=1 |
| 2026-10-01 | gpt-oss-120B (vLLM, Colab G4) | MTS_FS com tema | 299 | 0,41 | 0,59 | 0,60 | `--temas data/prompts_essaybr.csv`: título e texto motivador do tema (essay-br, lplnufpi) no prompt; até aqui o modelo nunca tinha visto a proposta. Total empata (0,592 contra 0,599). Discriminação melhora onde o tema importa: Pearson C3 0,43 para 0,49 (QWK C3 0,39 para 0,48), C2 0,45 para 0,48. Mas o modelo fica mais severo em tudo (viés total -91 para -153), o que derruba o QWK bruto de C2 e C4 |

### Feedback formativo (2026-10-01)

Smoke 1 (`mts_fb`, feedback junto com a nota, uma chamada por competência, 5 primeiras da
amostra de 300): 67 de 75 trechos citados literais. Problemas: o modelo corrige o erro do aluno
ao copiar o trecho (esconde justo o erro da C1), explicações de C1 sem sentido ("'não passava'
deveria ser 'não passava'"), o mesmo trecho citado nas 5 competências, e o feedback acompanha
a nota do modelo, que nessas 5 estava longe da humana. As 5 primeiras linhas da amostra são
as de menor nota (a amostra é salva em ordem de índice, que correlaciona 0,6 com a nota), por
isso o smoke de feedback passou a usar `data/amostra_feedback_30.csv`, embaralhada.

Smoke 2 (`gerar_feedback.py`, duas etapas: nota humana dada, uma chamada por redação para as
5 competências, com tema e campo `correcao`): 46 de 54 trechos literais. 6 dos 8 não literais
eram trechos do texto motivador citados como se fossem do aluno; o prompt de feedback passou a
levar só o título do tema. Pontos bons: correções úteis na C1 ("inciou" para "iniciou"),
reescritas da proposta da C5 com agente, ação, meio e efeito. Problemas que seguem:
explicações gramaticais da C1 às vezes erradas ou incoerentes com a correção, problema de C1
aparecendo na C3, e números inventados nas reescritas da C5 ("reduzir em 30% os impostos").

Smoke 3 (mesmas 5, só o título do tema no prompt e regra de não inventar números nas
correções): 51 de 54 trechos literais (94%), nenhum vindo do texto motivador. Os 3 restantes
são desvios mínimos (uma palavra trocada ou reordenada num trecho de 130 a 250 caracteres).
Números inventados nas correções caem de 8 para 3. Trechos repetidos entre competências
continuam (9 a 11 nas 5 redações), quase sempre a frase da proposta de intervenção citada na
C2, C3 e C5, o que é em parte natural. Aprovado para rodar as 30.

Rodada das 30 (`data/amostra_feedback_30.csv`), duas versões:

| Versão | Trechos literais | Números inventados | Trechos repetidos | Erro da nota usada |
|---|---|---|---|---|
| `feedback_humano_30` (nota humana) | 276/295 (94%) | 8 | 53 | 0 |
| `feedback_pred_30` (nota do 120b mts_fs) | 281/302 (93%) | 6 | 61 | 35 pts por competência |

Achado: o feedback quase não acompanha a nota dada. O número de problemas apontados é ~2 por
competência de 40 a 160 e só cai com 200 (0,4). Com nota humana 200 na C1 e na C4 (redação
902), o feedback aponta os mesmos problemas que com a nota 80 do modelo; com C5 humana 160
(redação 875), diz que a proposta "não especifica agente, ação, meio, efeito". O modelo escreve
a partir do próprio julgamento, não da nota recebida. É o critério "alinhamento com a nota"
da rubrica de avaliação; precisa de leitura humana para medir.

### QWK por competência (bruto), Flash Lite

| Modo | C1 | C2 | C3 | C4 | C5 |
|---|---|---|---|---|---|
| holístico | 0,29 | 0,46 | 0,38 | 0,29 | 0,33 |
| MTS | 0,29 | 0,50 | 0,41 | 0,40 | 0,33 |
| MTS2 | 0,21 | 0,56 | 0,40 | 0,38 | 0,32 |
| MTS_FS | 0,34 | 0,48 | 0,46 | 0,49 | 0,33 |
| MTS_FS2 | 0,35 | 0,47 | 0,43 | 0,48 | 0,35 |
| MTS_RR | 0,35 | 0,47 | 0,44 | 0,48 | 0,39 |
| MTS_LT | 0,38 | 0,47 | 0,43 | 0,48 | 0,35 |
| MTS_LT_RR | 0,39 | 0,47 | 0,44 | 0,48 | 0,39 |

O few-shot com âncoras (MTS_FS) move C1 (0,29 para 0,34) e sobe C3 e C4. O MTS_FS2 adiciona
checklist no C5 e finalmente tira o C5 do lugar (0,33 para 0,35), cortando o viés de nota 0.
No MTS_RR, MTS_LT e MTS_LT_RR só mudam C1 e/ou C5: as outras competências usam o mesmo
prompt do MTS_FS2 e vêm da mesma rodada (`reflect_revise.py --merge`).

Diferenças de 2026-09-29 testadas com bootstrap pareado (2000 reamostragens, mesmas redações
nos dois modos). Com ~290 redações, nenhuma diferença por competência é significativa, e o
total calibrado de todas as variantes cabe em 0,60 a 0,61. O MTS_FS2 segue como referência,
e o MTS_LT_RR é o melhor número bruto.

Correção de bug (2026-09-29): o merge somava as competências com `sum()` do pandas, que
ignora NaN, então redações com uma competência faltando entravam com total parcial. O
resultado do MTS_RR saiu primeiro como 0,58 calibrado por causa disso; o valor correto é
0,60. Agora o total fica vazio quando falta competência.

### LanguageTool no C1 (2026-09-29)

`lt_features.py` roda o LanguageTool 6.8 offline (pt-BR, precisa de Java 17) e conta, por
redação, desvios de gramática, ortografia, pontuação, maiúsculas e palavras confundidas.
Filtragem necessária por causa de artefatos do essay-br:
- o texto é uma lista de linhas; o `limpar()` deixa os separadores `', '` no meio, que viram
  erro de pontuação falso. O LanguageTool recebe as linhas juntadas com espaço;
- "espaço antes da pontuação" (366 ocorrências em 300 redações) vem do corpus ("país ."),
  assim como palavras repetidas ("anos anos", sobra das correções). Tipografia e repetição
  ficam de fora;
- ortografia só em palavra minúscula: siglas e nomes próprios (DUDH, Thrax, covid) não contam.

Com isso, a correlação de Spearman dos desvios por 100 palavras com o C1 humano vai de -0,27
para -0,37 (o LLM sozinho tem 0,43; a mistura de rankings dá 0,47). No prompt (`--modo
mts_lt`), o C1 sobe de 0,35 para 0,38 e o viés de C1 cai de -21 para -16, sem efeito
significativo no total.

Nota: o mesmo lixo do `limpar()` (separadores `', '`) vai no prompt de todos os modos. Não
foi mexido para não quebrar a comparação; limpar direito é um teste separado.

### Reflect-and-Revise da rubrica de C5 (2026-09-29)

`reflect_revise.py`: o modelo pontua o C5 de um fold de validação, lê a matriz de confusão
e até 12 casos com erro de 80 pontos ou mais (nota humana, nota dele, justificativa, trecho
final da redação) e reescreve a rubrica. 5 iterações, fica a versão de maior QWK de C5 na
validação. Fold de validação: 101 redações dos folds de treino, de 45 temas que não aparecem
no teste, sem as âncoras, com todas as faixas de C5 (`data/val_c5.csv`).

| Versão | QWK C5 validação | Viés C5 validação |
|---|---|---|
| v0 (rubrica do MTS_FS2) | 0,07 | -55 |
| v1 | 0,07 | -53 |
| v2 | 0,09 | -51 |
| v3 | 0,09 | -54 |
| v4 | 0,12 | -42 |
| v5 (escolhida) | 0,14 | -34 |

No teste (292 redações, já com a correção do bug do merge): C5 0,35 para 0,39, viés C5 de
-6 para +13, total bruto 0,59 para 0,62, total calibrado 0,605 para 0,597, Pearson 0,60 para
0,62.

Leitura:
- A validação não representou o teste. A mesma rubrica v0 dá 0,07 na validação e 0,35 no
  teste, com viés de -55 contra quase zero. O loop corrigiu um viés duro que só existia na
  validação, afrouxando a rubrica ("reflexões propositivas afastam o zero", "conte os
  elementos de forma inclusiva"), e no teste passou do ponto: 19 redações com C5 humano 80
  receberam 160 ou 200.
- A melhora na validação veio quase toda do viés, que a calibração já corrige. Com 100
  redações o erro padrão do QWK é de cerca de 0,1, então v0 a v5 estão dentro do ruído.
- A v5 tem um erro de conteúdo: diz "apologia aos direitos humanos" onde deveria ser
  desrespeito. Revisão automática de rubrica precisa de leitura humana antes de usar.
- Se repetir: validação sorteada dos mesmos temas do teste (sem as redações de teste) e
  critério de parada pelo QWK depois da correção de viés, não pelo QWK bruto.

## Cotas diárias das APIs gratuitas

Sempre conferir antes de rodar, os provedores mudam os limites com frequência.

### Gemini (Google AI Studio), limite por projeto Google

| Modelo | RPM | RPD | TPM | Redações/dia por conta |
|---|---|---|---|---|
| gemini-3.5-flash-lite | 15 | 500 | 250K | holístico ~500, MTS ~100 |
| gemini-3.x-flash (cheio) | 5 | ~20 (real ~13) | 250K | ~13 a 20 |
| gemma-4-31b-it | 30 | 14.400 | 16K | limitado pelo TPM, ~5/min |

Reset do RPD: meia-noite horário do Pacífico, cerca de 04:00 no horário de Brasília.
Temos 5 contas Google, então multiplicar por 5 (usar `--key-env GEMINI_API_KEY_1..5` e `--offset`).

### Groq, limite por conta

| Modelo | RPM | RPD | TPM | TPD | Redações/dia por conta |
|---|---|---|---|---|---|
| openai/gpt-oss-120b | 30 | 1.000 | 8K | 200K | holístico ~65 (limitado pelo TPD) |
| qwen/qwen3.6-27b | 30 | 1.000 | 8K | 200K | ~65 |

Temos 5 contas Groq. gpt-oss-120b holístico nas 300: cabe em 1 dia com as 5 contas.

### Fora

- Cerebras: sem crédito, trial de 5 dólares não foi provisionado para a organização.
- Maritaca (Sabiá): pedido de créditos acadêmicos feito no início de 2026, sem retorno.

## Gasto diário registrado

| Data | O que rodou | Chamadas por conta Gemini | Sobra estimada |
|---|---|---|---|
| 2026-09-08 | mts_fs 300 (60 x 5) + 2 smokes | ~350 (conta 1), ~300 (contas 2-5) | conta 1 ~150, contas 2-5 ~200 |
| 2026-09-09 | mts_fs2 300 (60 x 5), rodado pela colega na máquina dela | ~300 por conta | ~200 por conta |
| 2026-09-29 | Reflect-and-Revise (6 x 101 validação + 5 revisões + 300 teste só C5, colega) e mts_lt 300 só C1 | ~340 (conta 1, fez v0 e v5 e os smokes), ~220 (contas 2-5) | conta 1 ~160, contas 2-5 ~280 |

Regra: antes de disparar um run grande, somar o que já foi gasto no dia. mts_fs / mts_fs2
de 300 redações custam 300 chamadas por conta (60 redações x 5 competências). Cabe 1 run
por dia por lote de 5 contas. RPD zera ~04:00 BRT.

## Consumo por tipo de teste

- holístico: 1 chamada por redação
- MTS e MTS2: 5 chamadas por redação (uma por competência)
- self-consistency (ainda não usado): multiplica pelo número de amostras

## Próximos passos

Feitos: MTS_FS2 (2026-09-09), Reflect-and-Revise do C5 e LanguageTool no C1 (2026-09-29).
Os ganhos por prompt estão ficando dentro do ruído de 300 redações.

1. Ampliar o teste para mais redações (o essay-br tem ~1300 no fold 0), para conseguir
   separar diferenças de 0,02 a 0,03. Os modos de uma competência custam 1 chamada por
   redação, então cabe.
2. Limpar o texto que vai no prompt (separadores `', '` do `limpar()`), como teste isolado.
3. Trilha de feedback formativo: o modo MTS já gera justificativa por competência no campo
   `raw`. Avaliar com rubrica mais avaliação humana, usando o Banco de Redações da UOL como
   referência parcial.
4. gpt-oss-120B em modo MTS (Groq), para ver se o ganho do MTS vale para o modelo aberto.
5. Se repetir o Reflect-and-Revise: validação dos mesmos temas do teste e parada pelo QWK
   depois da correção de viés.

### Base na literatura (Qualis A/A1)

- Few-shot com âncoras: "Anchor is the key" (Studies in Educational Evaluation, 2026);
  "Specialists or Generalists?" (2026) reporta +26% de QWK com 2 exemplos por faixa.
- Reflect-and-Revise: "Automated Refinement of Essay Scoring Rubrics via Reflect-and-Revise"
  (CoNLL 2026), ganho de até +0,4 QWK sem treino.
- Features linguísticas: "Improve LLM-based AES with Linguistic Features" (arXiv 2502.09497).
- Teto realista: "Has AES Reached Sufficient Accuracy? QWK Ceilings from Classical Test
  Theory" (arXiv 2604.19131) e PROPOR 2026 (gap ao oráculo de 0,15 a 0,29 por trait).
  Alvo realista para o total: 0,65 a 0,68, não 0,73.
