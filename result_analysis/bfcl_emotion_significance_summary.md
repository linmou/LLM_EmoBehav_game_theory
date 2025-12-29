# BFCL Significance by Model (paired t across repeats)

Last Updated: 2025-10-01

Scope
- Per run: paired t-test on per-repeat means (emotion − neutral), df = n_pairs−1, alpha = 0.05 (two-sided).
- Aggregation: per model, report avg delta, avg t-stat, and fraction of runs marked significant.

## Llama-3.2-1B-Instruct
- Anger: Δ=-1.37 pp, t̄=inf, sig_rate=0.75
- Disgust: Δ=-1.37 pp, t̄=inf, sig_rate=0.75
- Fear: Δ=-1.37 pp, t̄=inf, sig_rate=0.75
- Happiness: Δ=-1.37 pp, t̄=inf, sig_rate=0.75
- Sadness: Δ=-1.37 pp, t̄=inf, sig_rate=0.75
- Surprise: Δ=-1.37 pp, t̄=inf, sig_rate=0.75

## Llama-3.2-3B-Instruct
- Anger: Δ=-10.34 pp, t̄=-18.08, sig_rate=0.67
- Disgust: Δ=-30.71 pp, t̄=-59.39, sig_rate=1.00
- Fear: Δ=-22.14 pp, t̄=-38.91, sig_rate=1.00
- Happiness: Δ=-14.83 pp, t̄=-13.54, sig_rate=1.00
- Sadness: Δ=-14.72 pp, t̄=-9.81, sig_rate=1.00
- Surprise: Δ=-18.98 pp, t̄=-23.42, sig_rate=1.00

## Phi-3.5-mini-instruct
- Anger: Δ=-0.44 pp, t̄=-0.35, sig_rate=0.00
- Disgust: Δ=-1.39 pp, t̄=-1.10, sig_rate=0.33
- Fear: Δ=-1.24 pp, t̄=-0.86, sig_rate=0.33
- Happiness: Δ=-2.33 pp, t̄=-0.92, sig_rate=0.33
- Sadness: Δ=-0.39 pp, t̄=-0.44, sig_rate=0.00
- Surprise: Δ=0.70 pp, t̄=-1.48, sig_rate=0.67

## Phi-4-mini-instruct
- Anger: Δ=1.73 pp, t̄=-3.85, sig_rate=0.75
- Disgust: Δ=2.00 pp, t̄=-1.91, sig_rate=0.75
- Fear: Δ=-0.78 pp, t̄=4.94, sig_rate=0.75
- Happiness: Δ=0.28 pp, t̄=7.34, sig_rate=0.75
- Sadness: Δ=0.20 pp, t̄=5.03, sig_rate=0.50
- Surprise: Δ=-0.57 pp, t̄=2.51, sig_rate=0.50

## Qwen2.5-0.5B-Instruct
- Anger: Δ=-4.51 pp, t̄=inf, sig_rate=0.50
- Disgust: Δ=-4.60 pp, t̄=inf, sig_rate=0.50
- Fear: Δ=-4.80 pp, t̄=inf, sig_rate=0.50
- Happiness: Δ=-4.02 pp, t̄=inf, sig_rate=0.50
- Sadness: Δ=-4.22 pp, t̄=inf, sig_rate=0.50
- Surprise: Δ=-4.18 pp, t̄=inf, sig_rate=0.50

## Qwen2.5-1.5B-Instruct
- Anger: Δ=0.66 pp, t̄=-8.37, sig_rate=0.50
- Disgust: Δ=-0.76 pp, t̄=inf, sig_rate=0.25
- Fear: Δ=-0.50 pp, t̄=-13.03, sig_rate=0.75
- Happiness: Δ=1.74 pp, t̄=-2.36, sig_rate=0.50
- Sadness: Δ=2.08 pp, t̄=5.22, sig_rate=0.50
- Surprise: Δ=-2.95 pp, t̄=inf, sig_rate=0.75

## Qwen2.5-3B-Instruct
- Anger: Δ=-8.83 pp, t̄=-20.83, sig_rate=1.00
- Disgust: Δ=3.50 pp, t̄=inf, sig_rate=1.00
- Fear: Δ=1.92 pp, t̄=4.80, sig_rate=0.50
- Happiness: Δ=-3.14 pp, t̄=-31.38, sig_rate=0.50
- Sadness: Δ=-9.99 pp, t̄=inf, sig_rate=1.00
- Surprise: Δ=-0.49 pp, t̄=-3.94, sig_rate=0.25

## Qwen3-0.6B
- Anger: Δ=-5.63 pp, t̄=-9.78, sig_rate=0.50
- Disgust: Δ=-6.31 pp, t̄=inf, sig_rate=1.00
- Fear: Δ=-0.81 pp, t̄=6.32, sig_rate=0.75
- Happiness: Δ=0.02 pp, t̄=-8.30, sig_rate=1.00
- Sadness: Δ=-8.64 pp, t̄=-17.20, sig_rate=1.00
- Surprise: Δ=-2.86 pp, t̄=1.55, sig_rate=0.75

## Qwen3-1.7B
- Anger: Δ=3.51 pp, t̄=inf, sig_rate=0.75
- Disgust: Δ=-0.06 pp, t̄=inf, sig_rate=0.50
- Fear: Δ=0.23 pp, t̄=-5.79, sig_rate=0.50
- Happiness: Δ=1.51 pp, t̄=inf, sig_rate=0.25
- Sadness: Δ=1.83 pp, t̄=inf, sig_rate=0.50
- Surprise: Δ=0.99 pp, t̄=inf, sig_rate=0.75

## Qwen3-32B-AWQ

## Qwen3-4B
- Anger: Δ=1.69 pp, t̄=inf, sig_rate=0.75
- Disgust: Δ=-1.41 pp, t̄=inf, sig_rate=1.00
- Fear: Δ=1.02 pp, t̄=-1.94, sig_rate=0.75
- Happiness: Δ=1.70 pp, t̄=inf, sig_rate=0.50
- Sadness: Δ=-0.80 pp, t̄=-1.85, sig_rate=0.25
- Surprise: Δ=0.37 pp, t̄=-0.88, sig_rate=0.50

## Qwen3-8B
- Anger: Δ=0.39 pp, t̄=inf, sig_rate=1.00
- Disgust: Δ=1.42 pp, t̄=4.16, sig_rate=0.00
- Fear: Δ=0.51 pp, t̄=1.51, sig_rate=0.00
- Happiness: Δ=-1.29 pp, t̄=-5.00, sig_rate=1.00
- Sadness: Δ=-1.16 pp, t̄=inf, sig_rate=1.00
- Surprise: Δ=1.03 pp, t̄=8.00, sig_rate=1.00

## gemma-3-1b-it
- Anger: Δ=0.08 pp, t̄=0.88, sig_rate=0.25
- Disgust: Δ=0.07 pp, t̄=0.63, sig_rate=0.00
- Fear: Δ=-0.02 pp, t̄=0.25, sig_rate=0.00
- Happiness: Δ=-0.05 pp, t̄=-0.50, sig_rate=0.00
- Sadness: Δ=-0.16 pp, t̄=-1.36, sig_rate=0.00
- Surprise: Δ=-0.11 pp, t̄=-0.51, sig_rate=0.00


References (run directories)
- Base: /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live
- Llama-3.2-1B-Instruct | live_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Llama-3.2-1B-Instruct_bfcl_live_multiple_20250927_183933
- Llama-3.2-1B-Instruct | live_parallel | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Llama-3.2-1B-Instruct_bfcl_live_parallel_20250927_225146
- Llama-3.2-1B-Instruct | live_parallel_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Llama-3.2-1B-Instruct_bfcl_live_parallel_multiple_20250927_233538
- Llama-3.2-1B-Instruct | live_simple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Llama-3.2-1B-Instruct_bfcl_live_simple_20250928_022825
- Llama-3.2-3B-Instruct | live_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Llama-3.2-3B-Instruct_bfcl_live_multiple_20250927_191031
- Llama-3.2-3B-Instruct | live_parallel | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Llama-3.2-3B-Instruct_bfcl_live_parallel_20250927_225522
- Llama-3.2-3B-Instruct | live_simple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Llama-3.2-3B-Instruct_bfcl_live_simple_20250928_023959
- Phi-3.5-mini-instruct | live_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Phi-3.5-mini-instruct_bfcl_live_multiple_20250927_193348
- Phi-3.5-mini-instruct | live_parallel | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Phi-3.5-mini-instruct_bfcl_live_parallel_20250927_225922
- Phi-3.5-mini-instruct | live_simple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Phi-3.5-mini-instruct_bfcl_live_simple_20250928_024739
- Phi-4-mini-instruct | live_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Phi-4-mini-instruct_bfcl_live_multiple_20250927_204952
- Phi-4-mini-instruct | live_parallel | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Phi-4-mini-instruct_bfcl_live_parallel_20250927_230437
- Phi-4-mini-instruct | live_parallel_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Phi-4-mini-instruct_bfcl_live_parallel_multiple_20250927_235857
- Phi-4-mini-instruct | live_simple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Phi-4-mini-instruct_bfcl_live_simple_20250928_025806
- Qwen2.5-0.5B-Instruct | live_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Qwen2.5-0.5B-Instruct_bfcl_live_multiple_20250927_163827
- Qwen2.5-0.5B-Instruct | live_parallel | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Qwen2.5-0.5B-Instruct_bfcl_live_parallel_20250927_222411
- Qwen2.5-0.5B-Instruct | live_parallel_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Qwen2.5-0.5B-Instruct_bfcl_live_parallel_multiple_20250927_231438
- Qwen2.5-0.5B-Instruct | live_simple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Qwen2.5-0.5B-Instruct_bfcl_live_simple_20250928_014158
- Qwen2.5-1.5B-Instruct | live_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Qwen2.5-1.5B-Instruct_bfcl_live_multiple_20250927_170318
- Qwen2.5-1.5B-Instruct | live_parallel | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Qwen2.5-1.5B-Instruct_bfcl_live_parallel_20250927_222749
- Qwen2.5-1.5B-Instruct | live_parallel_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Qwen2.5-1.5B-Instruct_bfcl_live_parallel_multiple_20250927_231827
- Qwen2.5-1.5B-Instruct | live_simple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Qwen2.5-1.5B-Instruct_bfcl_live_simple_20250928_015012
- Qwen2.5-3B-Instruct | live_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Qwen2.5-3B-Instruct_bfcl_live_multiple_20250927_171605
- Qwen2.5-3B-Instruct | live_parallel | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Qwen2.5-3B-Instruct_bfcl_live_parallel_20250927_223007
- Qwen2.5-3B-Instruct | live_parallel_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Qwen2.5-3B-Instruct_bfcl_live_parallel_multiple_20250927_232034
- Qwen2.5-3B-Instruct | live_simple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Qwen2.5-3B-Instruct_bfcl_live_simple_20250928_015717
- Qwen3-0.6B | live_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Qwen3-0.6B_bfcl_live_multiple_20250927_173130
- Qwen3-0.6B | live_parallel | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Qwen3-0.6B_bfcl_live_parallel_20250927_223353
- Qwen3-0.6B | live_parallel_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Qwen3-0.6B_bfcl_live_parallel_multiple_20250927_232318
- Qwen3-0.6B | live_simple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Qwen3-0.6B_bfcl_live_simple_20250928_020225
- Qwen3-1.7B | live_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Qwen3-1.7B_bfcl_live_multiple_20250927_175148
- Qwen3-1.7B | live_parallel | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Qwen3-1.7B_bfcl_live_parallel_20250927_223958
- Qwen3-1.7B | live_parallel_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Qwen3-1.7B_bfcl_live_parallel_multiple_20250927_232821
- Qwen3-1.7B | live_simple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Qwen3-1.7B_bfcl_live_simple_20250928_020940
- Qwen3-32B-AWQ | live_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Qwen3-32B-AWQ_bfcl_live_multiple_20250929_022046
- Qwen3-32B-AWQ | live_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Qwen3-32B-AWQ_bfcl_live_multiple_20250929_023139
- Qwen3-32B-AWQ | live_parallel | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Qwen3-32B-AWQ_bfcl_live_parallel_20250929_023557
- Qwen3-32B-AWQ | live_parallel_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Qwen3-32B-AWQ_bfcl_live_parallel_multiple_20250929_023821
- Qwen3-32B-AWQ | live_simple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Qwen3-32B-AWQ_bfcl_live_simple_20250929_024031
- Qwen3-4B | live_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Qwen3-4B_bfcl_live_multiple_20250927_181341
- Qwen3-4B | live_parallel | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Qwen3-4B_bfcl_live_parallel_20250927_224519
- Qwen3-4B | live_parallel_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Qwen3-4B_bfcl_live_parallel_multiple_20250927_233117
- Qwen3-4B | live_simple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Qwen3-4B_bfcl_live_simple_20250928_021346
- Qwen3-8B | live_simple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/Qwen3-8B_bfcl_live_simple_20250928_022005
- gemma-3-1b-it | live_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/gemma-3-1b-it_bfcl_live_multiple_20250927_215130
- gemma-3-1b-it | live_parallel | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/gemma-3-1b-it_bfcl_live_parallel_20250927_230947
- gemma-3-1b-it | live_parallel_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/gemma-3-1b-it_bfcl_live_parallel_multiple_20250928_010240
- gemma-3-1b-it | live_simple | /data/home/jjl7137/LLM_EmoBehav_game_theory/results/bfcl/live/gemma-3-1b-it_bfcl_live_simple_20250928_030328