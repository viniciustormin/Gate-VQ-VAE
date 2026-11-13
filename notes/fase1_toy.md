# Fase 1: toy 2D (escrito ANTES de rodar)

## Protocolo
- Dados: 8 gaussianas isotrópicas num anel (raio 2, sigma 0,1), 8192 amostras de treino e 2048 de teste.
- Modelo: MLP 2-64-64-2 no encoder e no decoder, d = 2, K = 16 codewords, init U(-1/K, 1/K).
- Estimadores: ste, gate_norm, gate_cb, dense_st. Seeds 0, 1, 2. Adam lr 3e-4, batch 128, 5000 passos.
- tau = 1 com `tau_scale: median` (mediana de ||z - e_k*||^2 no 1o batch, congelada). Fixado a priori,
  não será mudado depois de ver o dado. Os valores 5000 passos, raio e sigma são padrões meus.
- Saída em runs/khipu_toy/<estimator>_K16_tau1med_s<seed>/, com a trajetória do codebook a cada 50 passos.

## O que isto é (e não é)
Ilustração de mecanismo com 3 seeds num problema de brinquedo. NÃO testa H1 nem H2 do paper:
não há critério estatístico de decisão e nenhuma frase do paper deve sair daqui como evidência.

## Expectativas (NÃO VERIFICADO, só hipóteses de leitura)
1. ste e gate_cb entregam ao encoder exatamente o mesmo g; as latentes do encoder devem ser
   parecidas entre os dois e a diferença deve aparecer no movimento dos códigos.
2. gate_norm: o gradiente em z é (2/tau)(g.e_k*)(e_k* - ebar), que some quando p_k* -> 1
   (ebar -> e_k*). Espero campo de gradiente menor e mais "radial em torno do código" que o do ste.
3. dense_st mistura todos os códigos; espero gradiente denso e, possivelmente, mais códigos usados.
4. Nenhum limiar numérico. Reporto todos os números (média e desvio por estimador) e a figura.
   Se algum estimador der NaN ou colapsar, relato assim mesmo.

## Definição do campo de gradiente (figura)
Para um ponto z da grade no plano latente, o alvo é a amostra de teste cuja latente (saída do
encoder) é a mais próxima de z; o campo é -dL_rec/dz com L_rec = ||dec(z_q(z)) - x_alvo||^2 pelo
estimador, com o decoder e o codebook finais da seed 0. Todos os painéis usam a mesma escala de
setas. Escolha de definição minha, declarada na legenda da figura.

---
# Resultado (escrito DEPOIS de rodar; a seção acima não foi alterada)

12 execuções (4 estimadores x 3 seeds), 5000 passos, protocolo exatamente o acima.
Métricas no conjunto de teste ao fim; n = 3, desvio amostral. Fonte: runs/khipu_toy/*/done.json.

| estimador | MSE | % códigos usados | perplexidade |
|---|---|---|---|
| ste | 0,399 ± 0,120 | 47,9 ± 34,4 | 5,30 ± 2,64 |
| gate_norm | 0,336 ± 0,302 | 41,7 ± 7,2 | 5,77 ± 1,93 |
| gate_cb | 0,632 ± 0,275 | 31,3 ± 6,3 | 4,01 ± 0,86 |
| dense_st | 0,172 ± 0,113 | 47,9 ± 19,1 | 6,57 ± 1,93 |

Por seed (MSE): ste 0,445/0,490/0,263; gate_norm 0,607/0,392/0,010; gate_cb 0,636/0,355/0,906;
dense_st 0,042/0,241/0,234. A variação entre seeds é do tamanho das diferenças entre estimadores:
com n = 3 não há ordenação estimador-a-estimador que se sustente.

## Leitura contra as expectativas
1. "ste e gate_cb com latentes parecidas": NÃO confirmada. Na seed 0 as latentes do gate_cb vão
   a |z1| ~ 40 (ste: ~3), e o erro de quantização é 384 contra 1,4 (seeds 1 e 2: 808 e 1175
   contra 2,0 e 4,1). Os códigos do gate_cb quase não saem da origem (figura de trajetória).
2. "gate_norm com campo de gradiente menor": CONTRÁRIA. Norma mediana de g_z na seed 0:
   gate_norm 1,0e+01, ste 1,2e+00, dense_st 1,2e+00, gate_cb 1,7e+01 (mediana sobre a grade).
3. "dense_st usa mais códigos": não consistente entre seeds (68,8/43,8/31,2 % contra ste 25,0/31,2/87,5 %).

## Fato medido que afeta a Fase 2
O tau_eff congelado no 1o batch é minúsculo: 0,032 / 0,012 / 0,0011 (seeds 0/1/2; igual entre
estimadores da mesma seed, como esperado). Ele vem da mediana de ||z - e_k*||^2 no INÍCIO,
quando o encoder e o codebook (init U(-1/K, 1/K)) estão ambos perto de zero. Depois as
distâncias crescem (quant_err de 1e-3 a 1e3), então p = softmax(-d2/tau) tende a one-hot.
Que isso explique o comportamento do gate_cb e do gate_norm é HIPÓTESE, NÃO VERIFICADO (não
rodei nenhuma variação de tau; fazê-lo agora seria retunar depois de ver o dado).
