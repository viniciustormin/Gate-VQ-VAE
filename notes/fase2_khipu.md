# Fase 2: khipu em CIFAR-10 no Sauron (escrito ANTES de rodar)

## Protocolo
- CIFAR-10, conv_vqvae (128 canais, d = 64, 8x8 tokens por imagem), Adam 3e-4, batch 128,
  20k passos, avaliação a cada 2k, init uniforme idêntica entre estimadores, sem reinit de códigos mortos.
- (a) Piloto: gate_norm, gate_cb, dense_st; K = 1024; seed 0; tau em {0,1; 1; 10} x mediana (9 execuções).
- (b) Principal: ste, ste_ema, gate_norm, gate_cb, dense_st; K = 1024; seeds 0/1/2; tau escolhido
  pelo Vinícius depois do piloto (15 execuções).
- (c) K = 8192: ste, gate_cb, dense_st; seeds 0/1/2 (9 execuções).
- Decisão do Vinícius (13/11/2026): o piloto mantém o protocolo `tau_scale: median` como está
  (mediana de ||z - e_k*||^2 no 1o batch, congelada), sem mudar quando a mediana é calculada.

## Por que o tau do toy importa aqui
No toy o tau congelado saiu minúsculo (1e-3 a 3e-2) porque z e o codebook começam perto de zero
(notes/fase1_toy.md). Em CIFAR o encoder é convolucional e o init do codebook é o mesmo
U(-1/K, 1/K), então o mesmo efeito é PLAUSÍVEL, NÃO VERIFICADO.

## Expectativas do piloto (NÃO VERIFICADO; nenhum limiar a ser ajustado depois)
1. Se a saturação do softmax (p ~ one-hot) for o problema dos estimadores gate, tau maior (10 x
   mediana) deve dar mais códigos usados / perplexidade maior nos gate_* do que tau menor.
   Indicador por execução: log10(quant_err / tau_eff) ao fim (alto = saturado).
2. Se não houver diferença sistemática entre os tres tau, o piloto não discrimina e relato assim.
3. Reporto por execução: test_mse, test_psnr, codes_used_pct, perplexity, quant_err, tau_eff.
   O critério de escolha do tau é do Vinícius; esta lista é o que ele terá para decidir.

## Limitação declarada: seleção de tau no conjunto de teste
Só existe o conjunto de teste (10k imagens). Escolher tau pelo piloto nele infla de forma otimista
as métricas do experimento principal que usar o mesmo conjunto. Se o tau for escolhido por métrica
de teste, o paper deve dizer isso (ou o piloto deve usar uma fatia separada do treino: isso seria
desvio de protocolo, a decidir pelo Vinícius).

## Ordem de envio (cada passo só depois do anterior)
1. download.yaml: baixa o CIFAR-10 (job curto).
2. smoke.yaml: mede ms/passo e RAM no sauron; me mande o fim do log.
3. make_grid.py pilot --ms-per-step <medido>: gera os 3 jobs do piloto (e submit-pilot.sh).
4. Piloto -> o Vinícius escolhe tau -> make_grid.py main --tau X e k8192 --tau X.

## Tempo por execução (20k passos)
Medido SÓ na Mac (MPS), 400 passos de CIFAR-10 por execução, com 1 avaliação de 10k imagens
incluída (limite superior): K=1024 ste 34,6 ms/passo, gate_cb 39,2; K=8192 ste 52,6, gate_cb 98,4.
dense_st e gate_norm NÃO foram medidos (assumo perto do gate_cb, NÃO VERIFICADO). Em segundos por
execução na Mac: K=1024 ~11,5 a 13 min; K=8192 ~17,5 min (ste) e ~33 min (gate_cb).

| fase | execuções | tempo na Mac (estimativa, por job de 3 execuções) | jobs |
|---|---|---|---|
| piloto (K=1024) | 9 | ~40 min | 3 (um por estimador) |
| principal (K=1024) | 15 | ~40 min | 5 |
| K=8192 | 9 | ~1 h (ste), ~1,7 h (gate_cb, dense_st) | 3 |

No Sauron o tempo é DESCONHECIDO: a GPU é meia B200 (MIG), sem medida nesta sessão. O smoke
(jobs/khipu/smoke.yaml) mede isso em minutos; o `max_time` dos jobs do piloto usa 40 ms/passo da
Mac e está marcado "1ª ESTIMATIVA". Como o job retoma do checkpoint, estourar o max_time custa
só um reenvio.

Com 3 slots de 89 GB (cota de 268 GB), até 3 jobs rodam ao mesmo tempo, então cada fase leva
aproximadamente o tempo de 1 a 2 "ondas" de jobs.

## Medido no Sauron (smoke, job #3343, 13/11/2026)
GPU: NVIDIA B200 MIG 3g.90gb (89 GiB); torch 2.13.0+cu130 e torchvision 0.28.0 da imagem; commit 6f3d789.
200 passos por execução, com 1 avaliação de 10k imagens incluída (limite superior; a 1ª execução
também paga o aquecimento do CUDA):

| K | ste (ms/passo) | gate_cb (ms/passo) |
|---|---|---|
| 1024 | 14,9 | 16,5 |
| 8192 | 17,1 | 20,3 |

Na Mac (MPS) a mesma execução levou 34,6 a 39,2 ms (K=1024) e 52,6 a 98,4 ms (K=8192): o Sauron
foi ~2,3x mais rápido em K=1024 e ~2,6 a 4,8x em K=8192. Por execução de 20k passos isso dá
~5 a 7 min no Sauron (estimativa a partir de 200 passos; gate_norm e dense_st NÃO medidos).
Cada job de 3 execuções deve levar ~15 a 20 min.
RAM do host (cgroup): pico 1,6 GiB com as 4 execuções (download: 1,0 GiB), de 8 GiB pedidos
(20%, over-request). Dimensionado pela regra pico/0,7: `memory: 3g` no piloto.
O `max_time` do piloto ficou em 1h (25 ms/passo, arredondado para cima por causa dos estimadores
não medidos): é folga grande; se o piloto mostrar tempo real menor, baixar nas fases seguintes.

---
# Resultado do piloto (escrito DEPOIS de rodar; as seções acima não foram alteradas)

Jobs #3344, #3345 e #3346 (commit 908bfeb, B200 MIG 89 GB), 13/11/2026. K = 1024, seed 0 (n = 1),
20000 passos, métricas do conjunto de teste no passo final (sem early stopping). Fonte: `metrics.csv`
e `done.json` de runs/khipu/ no cluster. `tau_eff = tau x mediana`, com mediana = 0,3879 (igual nos
3 estimadores, como esperado). `qe` = quant_err.

| estimador | tau | MSE | PSNR | códigos usados % | perplexidade | qe | log10(qe/tau_eff) | melhor MSE nas 10 avaliações |
|---|---|---|---|---|---|---|---|---|
| gate_norm | 0,1 | 0,01104 | 19,57 | 13,1 | 21,0 | 4,27e-02 | 0,04 | 0,00902 |
| gate_norm | 1 | 0,00582 | 22,35 | 63,9 | 188,0 | 2,47e-02 | -1,20 | 0,00556 |
| gate_norm | 10 | 0,06171 | 12,10 | 0,7 | 1,0 | 7,15e-06 | -5,73 | 0,01309 |
| gate_cb | 0,1 | 0,00452 | 23,45 | 71,3 | 156,5 | 1,379 | 1,55 | 0,00444 |
| gate_cb | 1 | 0,00436 | 23,60 | 63,6 | 252,4 | 1,124 | 0,46 | 0,00428 |
| gate_cb | 10 | 0,00599 | 22,23 | 33,1 | 49,7 | 1,504 | -0,41 | 0,00571 |
| dense_st | 0,1 | 0,00351 | 24,54 | 92,3 | 434,8 | 0,4824 | 1,09 | 0,00345 |
| dense_st | 1 | 0,06184 | 12,09 | 0,3 | 1,0 | 2,14e-06 | -5,26 | 0,06182 |
| dense_st | 10 | 0,06184 | 12,09 | 0,2 | 1,0 | 1,18e-06 | -6,52 | 0,06184 |

## Leitura contra as expectativas escritas antes
1. "tau maior (10 x mediana) dá mais códigos usados nos gate_*": CONTRÁRIA. Com tau = 10, gate_norm
   e dense_st terminam colapsados em ~1 código (perplexidade 1,0, MSE ~0,062), e o gate_cb cai de
   63,6% (tau = 1) para 33,1%. dense_st também colapsa em tau = 1.
2. "sem diferença sistemática entre os três tau": CONTRÁRIA. A diferença é grande e o melhor tau
   depende do estimador: gate_norm melhor em 1, gate_cb em 1 (0,1 quase igual), dense_st só em 0,1.
3. A hipótese do toy (tau congelado minúsculo satura o softmax) NÃO se reproduz em magnitude: em
   CIFAR a mediana do 1o batch é 0,388, não 1e-3. Os colapsos ocorrem justamente com tau grande.
   Que um tau grande deixe p quase uniforme e o gradiente de dense_st/gate_norm pouco informativo é
   HIPÓTESE, NÃO VERIFICADO.

## Cuidados
- n = 1 seed: a ordem entre estimadores e a escolha do tau podem mudar com outras seeds.
- gate_norm com tau = 10 chegou a 0,0131 de MSE no melhor ponto e terminou em 0,0617: o colapso
  veio tarde (passo final usa o último checkpoint, sem escolher o melhor).
- Escolher tau por estas métricas é seleção no conjunto de teste, e a seed 0 do piloto reaparece
  na grade principal.

## Tempo e RAM medidos (a atualizar nas fases seguintes)
Cada execução levou 143 a 171 s (7,6 a 8,6 ms/passo); cada job de 3 execuções ~7 a 8 min. O smoke
(15 a 20 ms/passo) superestimou ~2x por incluir aquecimento. RAM do host: pico 1,5 a 1,6 GiB de
3 GiB (52 a 53%), dentro da faixa de 50 a 90%: manter `memory: 3g`. O `max_time` de 1h ficou
~7x acima do necessário: nas fases seguintes usar ~10 ms/passo (o gerador arredonda para 30m).

## Decisão do tau para as fases (b) e (c) (Vinícius, 13/11/2026)
tau POR ESTIMADOR, escolhido pelo melhor MSE final de cada um no piloto: gate_norm = 1, gate_cb = 1,
dense_st = 0,1. ste e ste_ema não usam tau; levam tau = 1 só no nome da pasta da execução
(`ste_K1024_tau1med_s0`). O mesmo mapa vale no K = 8192.
Desvio do pedido original (um tau único), declarado: um tau único faria dense_st colapsar (tau >= 1)
ou gate_norm piorar (tau = 0,1). Para o paper declarar: (1) tau escolhido por estimador, (2) escolha
feita no conjunto de teste, (3) com 1 seed, que reaparece na grade principal, e (4) o melhor tau do
dense_st está na borda da grade testada (0,1 é o menor valor): o ótimo pode estar abaixo.
Os 5 jobs da fase (b) e os 3 da (c) foram gerados com `make_grid.py` (jobs/khipu/main-*.yaml e
k8192-*.yaml, mais submit-main.sh e submit-k8192.sh).
