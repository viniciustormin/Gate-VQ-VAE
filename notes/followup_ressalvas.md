# Follow-up das duas ressalvas do paper (escrito ANTES de rodar)

Pedido do Vinícius (14/11/2026): olhar (A) a escolha do tau no conjunto de teste com 1 seed e (B) a seed 0
do dense_st em K = 8192 que não aprendeu, cuja causa não foi investigada. As rodadas originais
(piloto, grade principal, K = 8192) ficam intactas em runs/khipu; tudo abaixo entra ao lado.

## B. Por que o dense_st (K = 8192, tau 0,1, seed 0) não aprendeu

Fato (dos CSVs, não é hipótese): essa execução ficou em MSE 0,0618 (= prever a imagem média) desde o
passo 2000, com 69% dos códigos em uso e quant_err ~1e-6; nas execuções que falharam, log10(quant_err /
tau_eff) fica entre -4,7 e -6,5, e nas saudáveis entre -1,2 e +1,6.

Hipótese H-uniforme (formada DEPOIS de ver esses números, por isso eles não a confirmam): o tau_eff é
congelado no 1º batch, quando z tem escala ~0,6 e o codebook ~1/K. A commitment loss encolhe z para a
escala dos códigos; quando as distâncias ficam << tau_eff, p = softmax(-d2/tau_eff) vira quase UNIFORME,
ebar ~ média dos códigos, e o decoder recebe uma entrada quase constante: sem sinal para sair do platô.
(É o oposto da saturação one-hot que o toy sugeria.) tau maior => armadilha mais cedo: consistente com
os colapsos do piloto em tau >= 1, mas isso também já era conhecido.

Teste PROSPECTIVO (Mac/MPS, 3000 passos, sem usar o cluster), com `scripts/diag_dense.py`, que treina
com o mesmo código e registra a cada 250 passos: entropia normalizada de p (H(p)/log K, 1 = uniforme),
escala (RMS) de z e do codebook, quant_err, tau_eff, MSE de treino:
- D1 reproduz: dense_st K = 8192, tau 0,1, seed 0. Se o platô NÃO aparecer na MPS (MSE < 0,95 x
  baseline aos 3000 passos), o teste é inconclusivo por diferença de dispositivo e a investigação
  segue no cluster com o mesmo diagnóstico. Nada se conclui sobre a causa a partir da MPS nesse caso.
- Se D1 reproduz: H-uniforme prevê H(p)/log K próxima de 1 e z_rms colapsando abaixo do tau_eff^(1/2).
  Contra-evidência: platô com H(p)/log K << 1 (p concentrado): a hipótese está ERRADA.
- Intervenção D2: mesma execução com tau 0,03 e 0,01 (só muda o multiplicador). H-uniforme prevê que
  tau menor escapa do platô (MSE < 0,5 x baseline aos 3000 passos). Se nenhum escapar, a hipótese não
  explica a falha.
- Contraste: seeds 1 e 2 (saudáveis) com tau 0,1: devem ter H(p)/log K bem menor que a seed 0.
Nenhum limiar será ajustado depois do dado; o que sair, sai.

## A. Escolha do tau no conjunto de teste, com 1 seed

Procedimento correto: escolher tau num conjunto de VALIDAÇÃO separado do treino e só então medir no
teste. Plano (precisa do cluster; só submeto com o OK do Vinícius):
- Opção aditiva `data.holdout: N` (padrão 0, comportamento inalterado): separa N imagens do treino
  (split fixo, semente própria) e as usa no lugar do teste nas métricas. As rodadas ficam em
  runs/khipu_val e os números saem como "validação", nunca como teste.
- Varredura de tau para gate_norm, gate_cb e dense_st em K = 1024: {0,03; 0,1; 0,3; 1; 3; 10}, 3 seeds
  (54 execuções, ~2,5 min cada). A grade é mais larga que a do piloto de propósito: o melhor tau do
  dense_st estava na borda (0,1, o menor testado).
- REGRA de escolha, fixada agora: por estimador, o tau com menor MSE de VALIDAÇÃO médio sobre as 3 seeds,
  entre os que não falham em nenhuma seed (falha = MSE >= 0,95 x baseline). Empate (diferença < 1
  desvio da melhor): o tau menor. Os tau escolhidos vão para configs/khipu/main_tau_val.yaml.
- Se os tau escolhidos coincidirem com os atuais (gate_norm 1, gate_cb 1, dense_st 0,1), a grade
  principal vale como está, agora com a escolha legitimada. Se diferirem, os estimadores afetados são
  reexecutados no tau novo (K = 1024 e 8192, seeds 0 a 2) e a tabela antiga fica como rodada original.
- A grade principal usa o treino completo (50k) com o tau escolhido; a validação só escolhe o tau.

---
# Resultado de B (escrito DEPOIS de rodar; a seção acima não foi alterada)

Diagnóstico na Mac (MPS), 3000 passos, `scripts/diag_dense.py` (o laço de treino é idêntico ao do
train.py: teste tests/test_diag.py, pesos finais iguais bit a bit no toy em CPU). Fonte: runs/diag/*.csv.
Baseline = MSE de prever a imagem média = 0,06178; falha = MSE >= 0,0587.

| execução (dense_st) | MSE @250 | @1000 | @3000 | H(p)/log K @3000 | z_rms @3000 | resultado |
|---|---|---|---|---|---|---|
| K=8192, tau 0,1, seed 0 | 0,0623 | 0,0619 | 0,0615 | 1,000 | 0,0029 | preso no platô |
| K=8192, tau 0,1, seed 1 | 0,0620 | 0,0616 | 0,0615 | 1,000 | 0,0027 | preso no platô |
| K=8192, tau 0,1, seed 2 | 0,0619 | 0,0614 | 0,0131 | 0,774 | 0,0578 | saiu do platô entre 1000 e 3000 |
| K=8192, tau 0,03, seed 0 | 0,0622 | 0,0072 | 0,0037 | 0,411 | 0,0773 | saiu antes de 1000 |
| K=8192, tau 0,01, seed 0 | 0,0144 | 0,0058 | 0,0034 | 0,355 | 0,0517 | saiu antes de 250 |
| K=1024, tau 0,1, seed 0 | 0,0621 | 0,0075 | 0,0050 | 0,408 | 0,1080 | saiu antes de 1000 |
| K=1024, tau 1, seed 0 | 0,0622 | 0,0619 | 0,0619 | 1,000 | 0,0047 | preso no platô |

No cluster (B200), runs/khipu, K=8192, tau 0,1, MSE de teste a cada 2k passos: seed 0 fica em 0,0619
até 20k (nunca saiu); seed 1 em 0,0620, 0,0619, 0,0619, 0,0619 até 8k e termina em 0,0031; seed 2
em 0,0620 aos 2k, 0,0056 aos 4k e termina em 0,0028. K=1024: tau 0,1 já está em 0,0059 aos 2k; tau 1
e 10 ficam em 0,062 durante os 20k.

## Contra as previsões escritas antes
- D1 (reproduz na MPS): CONFIRMADA. MSE 0,0615 aos 3000 passos, H(p)/log K = 1,000.
- H-uniforme (H/log K ~ 1 e z_rms colapsado nas execuções presas): CONFIRMADA em todas as presas
  (H = 1,000; z_rms de 0,003 a 0,005, contra 0,06 no passo 1). Contra-evidência (platô com p concentrado):
  NÃO observada.
- D2 (tau menor escapa do platô): CONFIRMADA em 2 de 2 (tau 0,03 e 0,01, seed 0: MSE < 0,5 x baseline
  aos 3000 passos). n = 1 por intervenção, uma seed, MPS: indica, não prova.
- Contraste (as seeds saudáveis 1 e 2 teriam H(p)/log K bem menor que a seed 0 desde o início):
  REFUTADA. Todas as execuções começam no platô com p uniforme (H = 1,000 já no passo 1; MSE ~0,062
  aos 250 passos). Na MPS a seed 1 também ficou presa até 3000 passos. O que difere entre as seeds
  é QUANDO saem do platô, não se estão nele.

## Leitura (o que os dados sustentam e o que NÃO)
Sustentado: no dense_st todas as execuções partem de um platô em que p é uniforme e a reconstrução fica
no MSE da imagem média; a "falha" do K=8192 seed 0 é uma permanência no platô maior que o orçamento de
20k passos (a seed 1 só saiu depois de 8k). Um tau menor encurta o platô nos dois casos testados, e
um tau maior (>= 1 em K=1024) o estende além de 20k passos.
Mecanismo proposto, NÃO VERIFICADO: no início os codewords são quase idênticos (U(-1/K, 1/K), ~1e-4 em
K=8192) e p é uniforme por mais que tau seja pequeno em relação a ||z||^2; o gradiente
(2/tau) sum_j p_j (g.e_j)(e_j - ebar) é então da ordem da covariância dos códigos, minúscula, e a saída
do platô exige que uma diferença minúscula entre códigos cresça (tempo ~ log(1/escala inicial)/taxa,
com taxa ~ 1/tau). Isso explicaria por que K maior (escala inicial menor) e tau maior alongam o platô.
Não testei a escala de inicialização, que fica fora do protocolo.
Consequência para o protocolo: o melhor tau do dense_st DEPENDE do K (0,1 serve em K=1024 e não é
confiável em K=8192). Por isso a varredura de validação (A) passa a cobrir os dois K e a incluir tau 0,01.

---
# Extensão declarada do plano A (feita ANTES de rodar a varredura de validação)

Motivo: o resultado de B mostrou que o melhor tau do dense_st depende do K (0,1 serve em K = 1024 e
mantém o platô por >= 8k passos em K = 8192). Mudanças, todas anteriores a qualquer execução em
runs/khipu_val:
- A varredura cobre K = 1024 (gate_norm, gate_cb, dense_st) E K = 8192 (gate_cb, dense_st), não só 1024.
- A grade de tau passa de 6 para 7 valores: {0,01; 0,03; 0,1; 0,3; 1; 3; 10} (0,01 entra porque, em B,
  tau 0,01 escapou do platô antes de 250 passos).
- A regra de escolha NÃO muda (elegibilidade sem falhas e 3 seeds, menor MSE de validação médio,
  candidatos até +1 desvio do melhor, menor tau entre eles), mas agora vale por (estimador, K).
  Saída: configs/khipu/main_tau_val.yaml com `by_estimator_k`.
- Tamanho: 63 execuções em K = 1024 (9 jobs de 7) e 42 em K = 8192 (6 jobs de 7); ~2,6 min e ~3,8 min
  por execução (medidos na grade), ~5,4 h de GPU no total, 15 jobs de 89 GB.
- ste e ste_ema não usam tau e não entram na varredura. gate_norm não tem execuções em K = 8192 na
  grade principal e continua sem.
- Se os tau escolhidos para K = 8192 diferirem do tau usado na grade principal (que reaproveitou o
  tau de K = 1024), a grade de K = 8192 é reexecutada nesse tau; a rodada original fica como está.

---
# Resultado de A (escrito DEPOIS de rodar; as seções acima não foram alteradas)

Varredura de validação: 105 execuções (jobs #3396 a #3410, todos `succeeded` em 1 tentativa), 5000 imagens
do treino como validação (as métricas "test_*" são de VALIDAÇÃO), baseline de validação MSE 0,06153
(falha = MSE >= 0,05846). Tabela completa: paper/khipu/tables/table_tau_val.csv; escolha por
`scripts/select_tau.py`, regra pré-registrada, resultado em configs/khipu/main_tau_val.yaml.

| célula | tau escolhido | MSE de validação (média ± dp, n = 3) | tau da grade principal | coincide |
|---|---|---|---|---|
| gate_norm K=1024 | 0,1 | 0,01394 ± 0,00114 | 1 (0,01454 ± 0,00759) | NÃO |
| gate_cb K=1024 | 1 | 0,00431 ± 0,00000 | 1 | sim |
| dense_st K=1024 | 0,01 | 0,00339 ± 0,00015 | 0,1 (0,00339 ± 0,00020) | NÃO |
| gate_cb K=8192 | 1 | 0,00272 ± 0,00007 | 1 | sim |
| dense_st K=8192 | 0,03 | 0,00261 ± 0,00005 | 0,1 (0,00630 ± 0,00603) | NÃO |

## O que os dados dizem
- gate_cb: o tau da grade principal (1) foi o escolhido nos dois K. A ressalva "escolhido no teste" não
  mudou a escolha para este estimador.
- dense_st K=1024: tau 0,01; 0,03 e 0,1 empatam em 0,00339. A regra (menor tau dentro de 1 dp do melhor)
  escolhe 0,01; o tau 0,1 da grade principal está no empate e o MSE é o mesmo, mas a regra pede 0,01.
- gate_norm K=1024: tau 0,1 e 1 estão dentro de 1 dp um do outro; o MSE de validação do gate_norm é
  instável entre seeds (dp 0,0076 em tau 1). O MSE de validação em tau 1 (0,0145) é ~2,3x o MSE de teste da
  grade principal (0,0063); nos outros estimadores validação e teste coincidem (gate_cb 0,00431 contra
  0,00435; dense_st 0,00339 contra 0,00334). A causa NÃO foi investigada.
- dense_st K=8192: tau >= 0,3 falha nas 3 seeds; tau 0,1 não falha em nenhuma seed mas tem média e dp
  altos (0,0063 ± 0,0060), isto é, escapou do platô com tempos muito diferentes; tau 0,03 e 0,01 ficam em
  0,0026 e 0,0027 com dp ~1e-4. Consistente com o diagnóstico B (o tau 0,1 está na zona de platô longo).
- dense_st K=1024: tau >= 3 falha nas 3 seeds (como no piloto, que já mostrava tau 1 e 10 presos no teste;
  aqui tau 1 ficou em 0,0095 e não falhou na validação).

## Consequência (pela regra do plano): reexecutar as 3 células que diferem
gate_norm K=1024 em tau 0,1; dense_st K=1024 em tau 0,01; dense_st K=8192 em tau 0,03; 3 seeds, treino
completo (50k), teste de verdade. A seed 0 de gate_norm tau 0,1 (K=1024) já existe do piloto e é reutilizada
(mesma configuração). As execuções originais (tau 1 e tau 0,1) ficam em runs/khipu como rodada original. A
tabela principal do paper passa a usar configs/khipu/main_tau_val.yaml.

---
# Resultado das reexecuções (escrito DEPOIS de rodar; as seções acima não foram alteradas)

Jobs #3416 a #3418 (commit 0f6ef8b), todos `succeeded`: 8 execuções novas em runs/khipu (a seed 0 de
gate_norm tau 0,1 já existia do piloto e foi reutilizada). Tabelas: a original (tau escolhido no TESTE,
main_tau.yaml) ficou em paper/khipu/tables_tau_teste/; a nova (tau por VALIDAÇÃO, main_tau_val.yaml) está
em paper/khipu/tables/ e é a principal do paper, como o plano dizia.

MSE de teste final (x1e-3), média ± dp sobre 3 seeds, treino completo:

| célula | tau teste -> validação | tau escolhido no teste | tau escolhido por validação | falhas |
|---|---|---|---|---|
| dense_st K=8192 | 0,1 -> 0,03 | 22,58 ± 34,00 | 2,66 ± 0,05 | 1/3 -> 0/3 |
| dense_st K=1024 | 0,1 -> 0,01 | 3,34 ± 0,16 | 3,37 ± 0,01 | 0/3 -> 0/3 |
| gate_norm K=1024 | 1 -> 0,1 | 6,31 ± 0,50 | 13,51 ± 3,24 | 0/3 -> 0/3 |
| gate_cb K=1024 e 8192, ste, ste_ema | sem mudança de tau | iguais | iguais | 0/3 |

## O que isso mostra
- A "falha" do dense_st em K = 8192 era do tau (0,1 fica na zona de platô longo, como B indicou); com o tau
  da validação nenhuma seed falha e o dense_st passa a ser o melhor estimador também em K = 8192.
- gate_norm: o tau escolhido pela validação (0,1, por margem mínima: 0,01394 contra 0,01454 de tau 1,
  com dp 0,0076 em tau 1) deu MSE de TESTE pior que o tau 1 da grade original (13,51 contra 6,31). A
  validação (3 seeds, 45k imagens) foi ruidosa para este estimador. Seguimos a regra pré-registrada e
  reportamos os dois resultados; NÃO retunamos depois de ver o teste. O gate_norm é pior que o ste nos
  dois casos (razão 3,7 com o tau da validação, 1,7 com o do teste).
- Os resultados de gate_cb, ste e ste_ema não dependem da escolha.

---
# Extensão: gate_norm e ste_ema em K = 8192 (escrito ANTES de rodar; pedido do Vinícius, 14/11/2026)

Motivo: o pedido original (item c) só tinha ste, gate_cb e dense_st em K = 8192. H1 (gate_norm pior que ste)
só foi testada em K = 1024, onde o gate_norm é instável; e ste_ema ficou sem K = 8192 na tabela.

Plano (mesmas regras, nada muda no protocolo):
- gate_norm K = 8192: varredura de validação com os MESMOS 7 tau, 3 seeds, 5000 imagens do treino de
  validação (runs/khipu_val, 21 execuções); tau escolhido pela MESMA regra pré-registrada
  (scripts/select_tau.py); depois 3 execuções finais no treino completo, no tau escolhido (runs/khipu).
- ste_ema K = 8192: 3 execuções finais (seeds 0 a 2); não usa tau (tau = 1 só no nome da pasta).

Expectativas (NÃO VERIFICADO; nenhum limiar a ser ajustado depois):
1. H1 prevê MSE de teste do gate_norm MAIOR que o do ste (4,10 x 1e-3) em K = 8192. Se for menor, H1
   NÃO vale em K = 8192 e o paper diz que ela só vale em K = 1024.
2. ste_ema próximo do ste, como em K = 1024.
3. Como o dense_st em K = 8192, o gate_norm pode ter platô e depender muito do tau; a varredura mostra.
Se a varredura não tiver tau elegível (todas as seeds falham), relato isso e o gate_norm K = 8192 fica fora
da tabela principal, declarado.
