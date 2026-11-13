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
