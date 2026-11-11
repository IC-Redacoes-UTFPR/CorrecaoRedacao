# Correção automática de redações do ENEM com modelos de linguagem e geração de feedback formativo

Rascunho do relatório (2026-11-11). Equipe: Profa. Eliane Maria De Bortoli Fávero (orientação),
Prof. Ives Rene Venturini Pola (coorientação), Monica Paula Oliveira Mackert e Yuri Matsumoto Santos
(bolsistas). Os números vêm de `docs/log_experimentos.md`, que tem todas as tabelas e o histórico.

Pendente: avaliação humana do feedback (kit pronto em `avaliacao_humana/`), revisão das
orientadoras e decisão final sobre o que entra no texto.

## 1. Objetivo

Dar a nota de cada uma das cinco competências do ENEM (C1 a C5, de 0 a 200) e, principalmente,
gerar um feedback formativo que explique ao estudante por que recebeu cada nota e como melhorar.

## 2. Dados e protocolo

- **Corpus:** essay-br estendido, notas por competência; 6.485 redações de 151 temas na divisão
  por tema.
- **Divisão por tema (cross-prompt), resultado principal:** treino 4.573 redações (105 temas),
  validação 516 (17), teste 1.396 (29), temas disjuntos (`make_cross_prompt_split.py`). Mede a
  generalização para temas nunca vistos, que é o uso real. Época, pesos e calibração são escolhidos
  na validação; o teste é usado uma única vez por configuração.
- **Divisão oficial do essay-br:** reproduzida (`make_official_split.py`, semente 230,
  4.603/987/987), só para comparar com a literatura, que usa divisão aleatória (temas do teste
  vistos no treino).
- **Métricas:** QWK (kappa quadrático ponderado) da nota total, em faixas de 40 pontos, mais
  Pearson e QWK por competência. Diferenças testadas com bootstrap pareado (2.000 reamostras), por
  redação e por tema.

## 3. Modelos de nota

1. **gpt-oss-120b sem treino**, com exemplos âncora no prompt (few-shot), via vLLM.
2. **gpt-oss-20b com LoRA**, 3 épocas, uma chamada por redação. A nota de cada competência é a
   **nota esperada**: média das 6 notas possíveis ponderada pela probabilidade que o modelo dá a
   cada uma. Sem isso o modelo treinado usa só 2 notas por competência.
3. **BERTimbau base** com regressão (linha de base forte da literatura).
4. **Empilhamento por competência** (`empilhar.py`): regressão ridge sobre as notas previstas pelos
   modelos 2 e 3 e quatro medidas simples do texto (palavras, parágrafos, diversidade de vocabulário,
   palavras por frase), ajustada só na validação, com validação cruzada por tema.
5. **Calibração pela validação:** ajuste por quantil das notas totais, estimado na validação e
   aplicado ao teste.

## 4. Resultados da nota

### 4.1 Divisão por tema (teste, 1.396 redações de 29 temas não vistos)

| Modelo | QWK bruto | QWK calibrado (quantil) | Pearson | C1 | C2 | C3 | C4 | C5 |
|---|---|---|---|---|---|---|---|---|
| gpt-oss-120b sem treino | 0,45 | 0,60 | 0,63 | 0,21 | 0,46 | 0,47 | 0,31 | 0,25 |
| gpt-oss-20b LoRA, nota esperada | 0,66 | 0,69 | 0,69 | 0,49 | 0,53 | 0,49 | 0,54 | 0,47 |
| BERTimbau base | 0,68 | 0,71 | 0,72 | 0,42 | 0,51 | 0,46 | 0,58 | 0,56 |
| **Empilhamento por competência** | **0,72** | **0,74** | **0,74** | 0,47 | 0,51 | 0,48 | 0,63 | 0,56 |

O empilhamento supera o BERTimbau em +0,033 de QWK (IC95 +0,010 a +0,057, reamostrando temas) e o
gpt-oss-20b em +0,058 (IC95 +0,012 a +0,101). A validação também o escolhe (0,71 contra 0,70 do
BERTimbau), então a escolha não usou o teste. O ganho vem dos pesos por competência (C4 de 0,58
para 0,63), não das medidas de texto.

Tentativas sem ganho (detalhes no log): parágrafos marcados e redação longa como início mais fim
no BERTimbau, BERTimbau large, gpt-oss com parágrafos, 6 épocas (overfitting depois da 3ª),
contagem de erros do LanguageTool como feature extra (0,71 com, 0,72 sem).

### 4.2 Divisão oficial (comparação com a literatura)

| Modelo | QWK total |
|---|---|
| Amorim e Veloso (2017), features (Marinho et al., JIDM 2022) | 0,49 |
| Fonseca et al. (2018), features (Marinho et al., JIDM 2022) | 0,53 |
| BERTimbau base (Matsuoka, arXiv 2023, sem revisão por pares) | 0,79 |
| Nosso BERTimbau base | 0,74 (0,77 calibrado) |
| Nosso gpt-oss-20b LoRA, 4 épocas | 0,78 |

Na divisão oficial ficamos no nível do melhor resultado publicado. Na divisão por tema não
encontramos resultado publicado no essay-br. Barbosa et al. (JBCS 2025, PROPOR 2026) fazem
cross-prompt por competência em outro corpus (`aes_enem_dataset`, 385 redações), com QWK de 0,60 a
0,73 por competência e o melhor modelo escolhido em cada uma; não é comparável diretamente.

## 5. Feedback formativo

### 5.1 Geração

`gerar_feedback.py --por-competencia`, gpt-oss-120b, uma chamada por competência. O prompt recebe a
nota (humana ou a do sistema) e a descrição oficial do nível do ENEM daquela nota e do nível de
cima. Para cada competência: pontos fortes, até 3 problemas (trecho literal da redação, explicação
e correção) e ações para a próxima redação. Versão final (v3): parágrafos preservados, trechos de
3 a 15 palavras, a C5 cita a proposta de intervenção, e as competências 2 a 5 deixam a gramática
para a C1. Na apresentação ao aluno, os problemas que apontam o mesmo trecho em competências
diferentes aparecem juntos (`por_trecho()`): 634 problemas viram 479 trechos nas 60 redações.

### 5.2 Métricas automáticas (30 redações do teste; nota humana / nota do sistema)

| Versão | Alinhamento com a nota | Trechos repetidos | Gramática fora da C1 | Trechos literais |
|---|---|---|---|---|
| v1 por competência | -0,56 / -0,32 | 34% / 34% | 33% / 38% | 97% / 94% |
| v2 | -0,59 / -0,43 | 29% / 26% | 33% / 28% | 96% / 96% |
| **v3** | **-0,66 / -0,58** | 21% / 26% | **4% / 3%** | 95% / 96% |

Alinhamento: correlação de Spearman entre a nota e o número de problemas apontados (mais negativo
é melhor: nota alta, menos problemas). Em mais 30 redações a v3 repete o comportamento (-0,59 /
-0,45, gramática fora da C1 2% / 3%).

### 5.3 Juiz automático (prévia da avaliação humana)

`juiz_feedback.py`: um LLM de outra família (Gemini Flash Lite) julga cada feedback, na linha do
FeedEval (Chu et al., 2026). Em 240 feedbacks, as notas de 1 a 5 de especificidade, utilidade e
validade ficam todas entre 4,0 e 4,5 (efeito teto) e não separam as versões. A proporção de
problemas na competência certa separa: v3 84% / 79%, v1 77% / 75%, v2 69% / 72% (v3 menos v1:
+6,1 pontos percentuais, IC95 +0,9 a +11,4). Cerca de 85% dos problemas apontados procedem em todas
as versões. O juiz ainda não foi validado contra avaliadores humanos.

### 5.4 Avaliação humana (a fazer)

60 feedbacks v3 com a nota do sistema, em ordem embaralhada, dois avaliadores de Letras, rubrica de
4 critérios de 1 a 5 (especificidade, acionabilidade, correção, alinhamento com a nota). Concordância
entre avaliadores por QWK e acordo exato e adjacente, e correlação com o juiz automático
(`avaliacao_humana.py --concordancia`).

## 6. Discussão

- Na divisão por tema, o teto parece vir dos dados: entradas e modelos maiores não melhoraram, e o
  ganho veio de combinar modelos que erram de jeitos diferentes (o gpt-oss é melhor na C1, o
  BERTimbau na C4 e na C5).
- Para dar só a nota, um modelo pequeno (BERTimbau) já é competitivo; o LLM se justifica pelo
  feedback, que nenhum modelo de nota produz.
- O feedback segue a nota (alinhamento -0,58 com a nota do sistema) e cita trechos reais (95%).
  O principal erro restante é apontar um problema na competência errada (cerca de 1 em 5).

## 7. Trabalhos relacionados a citar

- Marinho, Anchiêta e Moura (JIDM 2022): corpus essay-br estendido.
- Matsuoka (arXiv 2023): BERTimbau no essay-br, divisão aleatória.
- Barbosa et al. (JBCS 2025; PROPOR 2026): cross-prompt por competência em português, outro corpus.
- Utino et al. (2025): LLMs zero-shot e one-shot no Essay-BR.
- Anchiêta et al. (WebMedia 2025): feedback zero-shot por competência do ENEM, avaliado com
  BERTScore contra feedback humano e por uma linguista. Não é reproduzível no essay-br, que não tem
  feedback escrito por corretores; a comparação no relatório fica qualitativa.
- Eltanbouly et al. (TRATES, 2025) e Do et al. (GAPS, 2025): cross-prompt por traço.
- Chu et al. (FeedEval, 2026) e Maier et al. (2026, revisão sistemática): avaliação de feedback
  gerado por LLM.
