# Notas do projeto — Stroke Prediction

Projeto de equipe de Redes Neurais Artificiais e Deep Learning (Insper, 2026.2): duas entregas sobre
o mesmo dataset — uma EDA que termina num pipeline de pré-processamento (25/10, 30% da nota) e o
treino de uma rede neural sobre o mesmo split e a mesma seed (13/11, 70%). O dataset é o Stroke
Prediction (Kaggle): 5.110 pacientes, 10 variáveis e um alvo binário de AVC com apenas 4,87% de
positivos — desbalanceamento que condiciona a métrica, a perda, o limiar e a própria forma de medir
se o resultado significa alguma coisa. Equipe: Gabriel Fernando Missaka Mendes, Eduardo Takei
Yaginuma e Luca Santana Feltrin.

Estas notas reúnem num lugar só as decisões das duas entregas, com a justificativa de cada uma e a
referência de onde ela vive no relatório e no código. Vêm em dois níveis: as **de base** (seção 4),
de prática padrão da disciplina, e as **do projeto** (seções 5 e 6), em que havia alternativa real em
disputa. Caminhos relativos a `docs/projects/`.

---

## 1. O projeto em uma página

| | |
|---|---|
| Dataset | Stroke Prediction Dataset (fedesoriano, Kaggle) — 5.110 linhas, 10 preditores |
| Tarefa | Classificação binária; alvo `stroke` |
| Desbalanceamento | 4,87% de positivos (249); ~19,5 negativos por positivo |
| Split | 80/20 estratificado, `random_state=42` → treino 4.088 (199 AVCs), teste 1.022 (50 AVCs) |
| Validação | `StratifiedKFold(5, shuffle=True, random_state=42)` sobre o treino |
| Métrica principal | Average precision (AP), fixada antes de treinar |
| Rede final | 23 → 16 ReLU → 1 logit, 401 parâmetros, Adam lr 3e-3, batch 128, 8 épocas |
| Resultado | AP de teste 0,2622 ± 0,0263 (5 seeds) |

**A tese, em uma frase:** idade carrega quase todo o sinal, e **a rede não mostra vantagem mensurável
sobre a regressão logística**. Quase toda decisão do projeto existe para tornar essa conclusão
defensável, e quase todo erro de leitura do relatório consiste em afirmar que a rede ganhou.

---

## 2. O que o enunciado exigia

Restrições, não escolhas da equipe — o relatório precisa mostrar que cumpriu cada uma:

- Rede feita à mão em PyTorch, Keras ou NumPy; `MLPClassifier` só valeria como baseline.
- Mesmo dataset, tarefa, split e seed da EDA; a validação sai de dentro do treino.
- **O teste é usado uma única vez, no step 5** — se tivesse ajudado em alguma decisão, o relatório
  seria obrigado a declarar e o step 5 valeria no máximo metade.
- Baseline trivial obrigatório (`DummyClassifier(strategy="most_frequent")`) e um linear; boosting
  opcional.
- Saída binária: 1 unidade com `BCEWithLogitsLoss`, devolvendo logits.
- Desbalanceamento por `pos_weight` na perda **ou** reamostragem só do treino.
- Métricas de teste: matriz de confusão, precisão/revocação/F1 por classe, macro F1, ROC-AUC e — por
  ser desbalanceado — curva precisão–revocação e PR-AUC.
- Limiar escolhido na validação e justificado pelo custo dos erros.
- ≥ 2 técnicas de regularização contra nenhuma e ≥ 8 configurações buscadas (foram 48).
- ≥ 3 seeds reportadas como média ± desvio (foram 5).
- Bônus de competição: não se aplica, o dataset é público e não é competição aberta.

**Pesos da entrega 2:** step 2 (a rede) e step 4 (regularização) valem 2,0 cada; steps 3 e 5, 1,5;
steps 1 e 6, 1,0; steps 0 e 7, 0,5.

**Reprodutibilidade:** cada entrega tem um `check_report.py` que, depois do `main.py`, confere que
toda tabela do relatório é idêntica à gerada pelo código e que os números citados no texto batem com
os salvos em `results/`.

---

## 3. Onde está cada coisa

| Fonte | O que tem |
|---|---|
| `classification/index.md` | A entrega principal: steps 0–7 |
| `eda/index.md` | A entrega 1: o pipeline que o Classification herda |
| `*/code/*.py` | A decisão como foi implementada — constantes, docstrings, comentários de rationale |
| `*/results/*.json`, `*.csv` | Os números gerados pelo código: a fonte da verdade numérica |

---

## 4. Decisões de base

As decisões abaixo seguem a prática padrão da disciplina e não foram disputadas dentro do projeto.
Ficam registradas porque são o chão sobre o qual as seções 5 e 6 se apoiam: várias delas aparecem lá
de novo, em versões com alternativa em disputa.

**B1 · As features numéricas são padronizadas antes de entrar na rede.** *Por quê:* idade vai de 0 a
82, glucose de 55 a 272. Sem padronizar, o gradiente dá passos desproporcionais por direção e a
convergência passa a depender de um learning rate que sirva às duas escalas ao mesmo tempo.
**Onde vive:** EDA 4A-4, Tabela 15; `eda/code/preprocessing.py:63–71`.

**B2 · As categóricas entram como one-hot, não como inteiros ordenados.** *Por quê:* codificar
`work_type` como 0, 1, 2, 3 inventaria uma ordem e uma distância que não existem, e a rede as leria
como grandeza. **Onde vive:** EDA 4A-3; `eda/code/preprocessing.py:72–75`.

**B3 · Faltantes numéricos são imputados pela mediana, não pela média.** *Por quê:* BMI e glucose são
assimétricos com cauda longa (E4); a média é puxada pelos extremos, a mediana não.
**Onde vive:** EDA 4A-1; `eda/code/preprocessing.py:64, 68`.

**B4 · O split é estratificado *e* embaralhado.** *Por quê:* o arquivo vem ordenado pelo alvo — os 249
positivos são as primeiras linhas —, então sem embaralhar o teste não teria positivo nenhum; a
estratificação mantém os 4,87% nas duas partes. **Onde vive:** EDA 1C–1D, Figura 1;
`eda/code/preprocessing.py:42–50`.

**B5 · ReLU nas camadas ocultas.** *Por quê:* derivada constante na parte positiva, sem a saturação de
sigmoid e tanh que faz o gradiente encolher camada após camada — e é barata de calcular.
**Onde vive:** step 2, Tabela 3; `classification/code/network.py:37`.

**B6 · O treino usa mini-batches, não o conjunto inteiro de uma vez.** *Por quê:* um passo de
otimização por batch dá dezenas de atualizações por época, enquanto o batch cheio daria uma só; e o
ruído de amostragem do batch é parte do que evita que o treino se cole a um mínimo qualquer.
**Onde vive:** step 3; `classification/code/network.py:92–93`. (C6 trata do tamanho 128.)

**B7 · As linhas são embaralhadas a cada época (`shuffle=True` no `DataLoader`).** *Por quê:* sem
isso a rede veria sempre a mesma sequência de batches, e a ordem do arquivo viraria estrutura do
treino. **Onde vive:** `classification/code/network.py:92–93`.

**B8 · A perda é entropia cruzada binária, não erro quadrático.** *Por quê:* a saída é uma
probabilidade, e a entropia cruzada penaliza a confiança errada de forma crescente; combinada com a
sigmoid, dá um gradiente que não satura quando a predição está muito errada — que é exatamente onde o
erro quadrático empaca. **Onde vive:** step 2; `classification/code/network.py:86`.

**B9 · O modelo passa para `model.eval()` antes de prever.** *Por quê:* dropout é comportamento de
treino; na avaliação a rede precisa usar todas as unidades. É pela mesma razão que as curvas de perda
de treino e validação são medidas com dropout desligado — só assim são comparáveis.
**Onde vive:** step 3, Figura 1; `classification/code/network.py:61–65, 107`.

**B10 · Uma época é uma passagem completa pelos dados de treino; o teto é 300 e quem decide o fim é o
early stopping.** *Por quê:* o número de épocas é hiperparâmetro. Fixá-lo no olho arrisca parar antes
de aprender ou seguir até decorar o treino. **Onde vive:** step 3, Tabela 4;
`classification/code/network.py:17–19`. (C7 trata do critério de parada e da paciência.)

**B11 · Validação cruzada em 5 folds, não um único conjunto de validação.** *Por quê:* com 199
positivos no treino, um hold-out único deixaria ~40 positivos para medir tudo, e a estimativa
dependeria de quais 40 caíram lá. Com 5 folds, cada linha é usada uma vez para validar.
**Onde vive:** steps 0–1, Tabela 2; `classification/code/common.py:66–68`.

**B12 · Adam em vez de SGD puro.** *Por quê:* Adam adapta o tamanho do passo de cada parâmetro, o que
ajuda quando as entradas são tão diferentes quanto idades padronizadas e flags 0/1.
**Onde vive:** step 3; `classification/code/network.py:89–90`.

**B13 · As seeds são fixas e declaradas.** *Por quê:* o split, a inicialização dos pesos, as máscaras
de dropout e a ordem dos batches são todos aleatórios; sem seed, dois runs dão números diferentes e
nenhuma tabela do relatório seria reproduzível. **Onde vive:** "Seeds", no início do Classification;
`classification/code/network.py:82`, `classification/code/common.py:42`.

**B14 · O desempenho é reportado por matriz de confusão e por precisão/revocação de cada classe, não
só por uma métrica agregada.** *Por quê:* a matriz mostra *qual* dos dois erros o modelo comete; com
4,87% de positivos, dois modelos com a mesma acurácia podem errar de formas opostas.
**Onde vive:** step 5, Tabelas 8–9, Figura 5.

---

## 5. Decisões da EDA

As decisões abaixo não ficam na entrega 1: o Classification herda o split, o pipeline e as regras que
a EDA deixou escritas.

**E1 · A métrica foi escolhida pelo desbalanceamento, não pelo costume.**
*Descartado:* acurácia. **Por que não é óbvio:** um modelo que nunca prevê AVC acerta 95,13%; a
acurácia premia o modelo inútil, e o ROC-AUC esconde falsos alarmes atrás do volume de verdadeiros
negativos. **Onde vive:** EDA seção 0 e 1C; Classification step 0.

**E2 · `smoking_status = Unknown` é mantido como categoria própria.** *Descartado:* imputar, ou tratar
como ruído. **Por que não é óbvio:** é um faltante disfarçado (30,22% das linhas), mas **não é
aleatório** — 498 de 554 linhas `children` são `Unknown`. Informativo, logo é categoria.
**Onde vive:** EDA 1B e 4A.

**E3 · BMI: imputação por mediana + indicador de faltante.** *Descartado:* remover as linhas; imputar
sem indicador. **Por que não é óbvio:** a taxa de AVC é ~5× maior quando o BMI falta (21,76% × 4,13%),
o que faz o indicador parecer uma feature forte e, ao mesmo tempo, levanta a suspeita de ser artefato
de coleta. A EDA deixa a regra escrita e o Classification a executa (C1). **Onde vive:** EDA 4A-1,
Figura 9; `eda/code/preprocessing.py:79`.

**E4 · `log1p` em glucose e BMI, não em age.** *Descartado:* padronizar tudo igual. **Por que não é
óbvio:** o critério é assimetria > 1 (glucose 1,56; BMI 1,12; age não). E o log **não** resolve a
bimodalidade da glucose — ele conserta a cauda, a estrutura de dois modos continua lá e vira problema
do modelo. **Onde vive:** EDA 2A e 4A-1, Tabela 14; `eda/code/preprocessing.py:29–31, 67–71`.

**E5 · Nenhuma linha é removida por outlier, embora as cercas sejam calculadas.** *Descartado:* aplicar
a regra 1,5×IQR. **Por que não é óbvio:** aplicá-la apagaria o segundo modo da glucose, que é
justamente o grupo de maior risco (12,94% de AVC acima do vale de 175 mg/dL). **Onde vive:** EDA 4A-2,
Tabela 13.

**E6 · O pipeline é um `ColumnTransformer` ajustado só nas linhas de treino.** **Por que importa:** o
teste é apenas transformado, e a prova disso está nos parâmetros ajustados (Tabela 15). Saída: **24
colunas na EDA**. **Onde vive:** EDA 4C; `eda/code/preprocessing.py:53–81`.

---

## 6. Decisões do Classification

**C0 · Métrica, critério de sucesso e custo do erro foram fixados e commitados *antes* do primeiro
treino.** *Descartado:* definir o critério depois de ver os resultados. **Por que não é óbvio:** o
critério escrito (AP média de 5 seeds ≥ 15% acima do melhor baseline **e** diferença maior que o desvio
entre seeds) é o que impede o relatório de mover a trave — e é exatamente por isso que ele acaba sendo
atendido por um resultado que a conclusão, mesmo assim, rejeita (C12). **Onde vive:** step 0, commit
`66f89ca`; `classification/code/common.py:36–42`.

**C1 · O indicador de BMI é removido — para todos os modelos.** *Descartado:* mantê-lo (ele aumenta o
AP de validação da rede em 0,0247 e da logística em 0,0308). **Por que não é óbvio — três camadas:**
(a) a regra da EDA diz que *qualquer* ganho torna o indicador suspeito de ser artefato de coleta;
(b) os ganhos são menores que o desvio entre folds, então nem são sólidos; (c) **a remoção muda qual
baseline é o melhor na validação** — com indicador seria a logística, sem ele é o gradient boosting —
e é contra esse baseline que o critério do step 0 é medido. Decidir por princípio aqui muda o resultado
do teste depois. **Onde vive:** step 0, Tabela 1; `classification/code/starting_point.py:42`,
`classification/code/validation.py:23–27`. A saída do pipeline cai para **23 colunas**.

**C2 · Três baselines: trivial, linear e boosting (opcional).** **Por que não é óbvio:** o trivial não
é enfeite — ele marca o piso (AP = prevalência = 0,0487, ROC-AUC = 0,5), e é contra esse piso que o AP
é lido. Os dois aprendidos usam `class_weight="balanced"`, que é o espelho do `pos_weight` da rede: a
comparação só é justa se o desbalanceamento for tratado igual nos três. **Onde vive:** step 1,
Tabela 2; `classification/code/baselines.py:12–19`.

**C3 · Uma unidade de saída devolvendo *logit*, com `BCEWithLogitsLoss`.** *Descartado:* duas unidades
com softmax; sigmoid seguida de `BCELoss`. **Por que não é óbvio:** a loss aplica a sigmoid
internamente de forma numericamente estável, e é ela que aceita `pos_weight`. A sigmoid explícita só
aparece na hora de prever. **Onde vive:** step 2; `classification/code/network.py:41–49, 86`.

**C4 · `pos_weight` = negativos/positivos, calculado *dentro* de cada fold.** *Descartado:* um valor
fixo calculado no treino inteiro; reamostragem. **Por que não é óbvio:** calcular dentro do fold evita
que a proporção do conjunto completo vaze para a validação (19,44 a 19,57 entre folds, 19,54 no treino
cheio) — diferença numericamente mínima, mas metodologicamente é o ponto. E esse mesmo `w` reaparece
como o fator que descalibra as probabilidades (C10). **Onde vive:** step 2;
`classification/code/network.py:85–86`.

**C5 · He (Kaiming) nas camadas ocultas, Glorot (Xavier) na saída, viés zero.** *Descartado:* uma
inicialização só. **Por que não é óbvio:** a escolha segue a ativação — He preserva a variância através
de ReLU; a camada de saída não tem ReLU depois, então usa Xavier. Viés 0 é coerente com `pos_weight`:
com as classes pesando igual, logit ≈ 0 (chance par) é o ponto de partida certo. **Onde vive:** step 2,
Tabela 3; `classification/code/network.py:24–45`.

**C6 · Batch de 128.** *Descartado:* batches menores. **Por que não é óbvio:** o tamanho é escolhido
pela prevalência, não pela memória — a 4,87%, 128 linhas trazem ~6 positivos, e um batch sem nenhum
positivo é raro (0,17%). **Onde vive:** step 3; `classification/code/network.py:16`.

**C7 · Early stopping na *loss* de validação, paciência 30, melhores pesos restaurados.**
*Descartado:* parar pelo AP de validação; paciência curta; usar a última época. **Por que não é
óbvio:** com 39–40 positivos por fold a curva balança, então paciência curta pararia num solavanco; e o
código registra de propósito o AP máximo, a época desse máximo e o AP final, para comparar o que a
*loss* e o *ranking* dizem sobre quando parar — duas coisas que não coincidem. **Onde vive:** step 3,
Tabela 4, Figura 1; `classification/code/network.py:17–19, 117–125`,
`classification/code/search.py:155–159`.

**C8 · Oito variantes de regularização contra nenhuma; dropout não ajuda.** *Descartado:* assumir que
dropout sempre ajuda. **Por que não é óbvio:** a leitura não é "dropout é ruim", é "dropout é técnica
para rede com capacidade sobrando" — com 32 e 16 unidades não há redundância para desligar. O weight
decay 1 é o único que estreita o gap sem derrubar o AP; o 10 faz as duas curvas se encontrarem alto
(underfitting). **Onde vive:** step 4, Tabela 5, Figura 2; `classification/code/search.py:17–29`.

**C9 · A rede final treina por 8 épocas — a mediana das épocas restauradas nos folds.**
*Descartado:* treinar a rede final com early stopping. **Por que não é óbvio:** no retreino final não
sobra conjunto de validação para parar; a mediana (de 12, 6, 6, 29, 8) é o substituto, e a verificação
de que isso funcionou só aparece no step 5. **Onde vive:** step 4 final, step 5;
`classification/code/search.py:212`.

**C10 · O limiar sai da validação e do custo 20:1, não do 0,5.** *Descartado:* 0,5; e também o p\* que
a teoria prediz. **Por que não é óbvio — a reconciliação:** o limiar escolhido nos dados é 0,4732,
enquanto o custo 20:1 com probabilidade calibrada preveria p\* = 1/(1+20) = 0,0476. Parecem
incompatíveis; só se encontram quando se desfaz o fator `pos_weight` (w = 19,54): dividindo as *odds*
de 0,4732 por w sai 0,0439 ≈ 0,0476. Ou seja, **as probabilidades da rede não são riscos** — e a curva
de custo é plana perto do mínimo, então o custo assumido importa mais que o corte exato. Cada modelo
ganha seu próprio limiar pela mesma regra (a do boosting é 0,0188, escala diferente). **Onde vive:**
step 5, Tabela 7, Figura 3; `classification/code/evaluate.py:28–34, 150–156`.

**C11 · O teste é lido uma vez, no step 5, com 5 seeds.** *Descartado:* olhar o teste antes; reportar
uma seed. **Por que não é óbvio:** o enunciado penaliza o step 5 pela metade se o teste tiver ajudado
em alguma decisão, e o relatório declara explicitamente que nada mudou depois da primeira leitura. As
seeds mudam **só a inicialização e a ordem dos batches** — não os dados, não a configuração, não o
limiar. **Onde vive:** step 5, Tabela 10; `classification/code/common.py:42`,
`classification/code/evaluate.py:165–167`.

**C12 · O critério é atendido e, mesmo assim, a conclusão é "não valeu a pena".** **Por que não é
óbvio — é o ponto central do trabalho:** a rede fica 26,37% acima do gradient boosting (o baseline que
a *validação* escolheu), com diferença maior que o desvio entre seeds: critério cumprido como escrito.
Mas (a) a escolha do baseline foi um quase-empate na validação; (b) no teste o vice, a regressão
logística, faz 0,2675 — **1,97% acima** da média da rede; (c) o bootstrap pareado (2.000 reamostragens,
mesmas linhas para todos os modelos) dá intervalo [−0,0418; 0,1581] para a própria diferença que cumpre
o critério — **contém 0**; (d) o AP de teste só é conhecido a ±0,1, porque repousa sobre 50 AVCs.
Cumprir um critério não é a mesma coisa que ter uma vantagem mensurável. **Onde vive:** step 5 final e
step 7; `classification/code/evaluate.py:104–126, 218–233`.

**C13 · A análise de erro usa os subgrupos que a EDA apontou — e `gender` como controle negativo.**
*Descartado:* escolher subgrupos pelo resultado. **Por que não é óbvio:** `gender` entra justamente
porque a EDA o achou **não** associado (Cramér's V 0,011); é controle, não hipótese. O achado que
importa é que o erro se parte em duas direções opostas por idade: a 60+ a rede alarma quase todo mundo
(96,15% dos não-AVCs) e não perde nenhum AVC; entre 18 e 59 ela perde 10 dos 11 AVCs. Quem é
prejudicado pelo erro é o paciente jovem que tem AVC. **Onde vive:** step 6, Tabela 11, Figura 6;
`classification/code/errors.py:26–37`.

**C14 · A conclusão contesta o próprio resultado, e as limitações são declaradas.** **Por que importa:**
a busca foi desigual (48 configurações para a rede, defaults de biblioteca para os baselines) — o que
favorece a rede, e ela ainda assim não ganha. O uso responsável é a consequência prática: a rede só
poderia *acrescentar* pacientes a uma lista de acompanhamento, nunca tirar alguém dela, porque o
silêncio dela é menos confiável exatamente onde o AVC é menos esperado. **Onde vive:** step 7.

---

## 7. Pares de números fáceis de trocar

Cada par abaixo mede **coisas diferentes**. Trocar um pelo outro muda a afirmação.

| Par | O que cada um é |
|---|---|
| 24 × 23 colunas | saída do pipeline na EDA × no Classification (indicador de BMI removido) |
| 0,0487 × 0,0489 | prevalência no treino × no teste (= AP do baseline trivial em cada um) |
| 1.313 × 401 | parâmetros da rede do step 2 (32-16) × da rede escolhida (16) |
| 0,2622 × 0,2997 | AP de teste: média das 5 seeds × a seed 42 (a mais sortuda, a das figuras) |
| 0,2622 × 0,2675 × 0,2075 | rede (média) × regressão logística × gradient boosting, no teste |
| 0,4732 × 0,0476 × 0,0439 | limiar escolhido × p\* teórico × limiar com o fator `pos_weight` desfeito |
| 0,0188 × 0,4280 × 0,4732 | limiares do boosting × da logística × da rede (escalas diferentes) |
| 15% × 26,37% × −1,97% | ganho exigido × ganho sobre o boosting × diferença para a logística |
| 199 × 50 | AVCs no treino × no teste (os dois limites de tudo que é estimado) |
| 39–40 × 817–818 | positivos × linhas por fold de validação |
| 19,54 × 20 | `pos_weight` (razão negativos/positivos) × custo de um AVC perdido em falsos alarmes |
| 300 × 30 × 8 | teto de épocas × paciência do early stopping × épocas do retreino final |
| 0,834 × 0,8413 × 0,8440 | ROC-AUC só da idade × da rede × da logística |
| 82,72% × 90,91% × 96,15% | taxa de erro dos 60+ × AVCs perdidos entre 18–59 × falsos alarmes entre os 60+ |
| 5 × 3 seeds | seeds efetivamente usadas × o texto do *template* na linha 6 da tabela-resumo (são 5) |

---

## 8. Leituras erradas comuns

Cada afirmação abaixo parece certa e não é. O relatório refuta todas, no ponto indicado.

- *"O critério foi atendido, então a rede é melhor."* — ignora C12 inteiro.
- *"AP alto significa boa probabilidade de AVC para o paciente."* — AP mede *ranking*; as
  probabilidades estão infladas por `pos_weight` (C10).
- *"Dropout sempre ajuda; foi usado errado."* — a leitura correta é capacidade sobrando (C8).
- *"O indicador de BMI foi removido porque não ajudava."* — ele **ajudava**; foi removido por
  princípio, e isso custou AP de validação (C1).
- *"O limiar 0,4732 contradiz o custo 20:1."* — só até desfazer o `pos_weight` (C10).
- *"Bastava remover os outliers de glucose."* — apagaria o grupo de maior risco (E5).

---

## 9. Notas de fonte

- **Peso das entregas.** A página de cronograma do curso diz 30%/70%; a página da entrega 2 diz "60% da
  nota do Projeto". As fontes divergem; o relatório não depende disso.
- **"3 seeds" na tabela-resumo.** A linha 6 da tabela-resumo do Classification diz *"(mean ± std, 3
  seeds)"* porque é o texto do template do enunciado. Os números ali são de **5** seeds.
- **O dataset não tem data.** Nada no arquivo diz quando cada medida foi tomada. É a origem de E3 e
  C1, e da limitação declarada no step 7: o modelo classifica o rótulo registrado, não prevê AVC futuro.
