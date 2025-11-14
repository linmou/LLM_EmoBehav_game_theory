# BFCL (Live) Emotion Impact by Model

Last Updated: 2025-10-01

Scope
- Inputs: all `summary_overall.csv` under `/data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live`.
- Metric: mean_of_means per emotion; deltas vs neutral (pp); averages across BFCL live categories per model.

## Llama-3.2-1B-Instruct
- Neutral baseline (avg across categories): 1.37%
- Anger: -1.37 pp
- Disgust: -1.37 pp
- Fear: -1.37 pp
- Happiness: -1.37 pp
- Sadness: -1.37 pp
- Surprise: -1.37 pp

## Llama-3.2-3B-Instruct
- Neutral baseline (avg across categories): 33.04%
- Anger: -10.34 pp
- Disgust: -30.71 pp
- Fear: -22.14 pp
- Happiness: -14.83 pp
- Sadness: -14.72 pp
- Surprise: -18.98 pp

## Phi-3.5-mini-instruct
- Neutral baseline (avg across categories): 29.87%
- Anger: -0.44 pp
- Disgust: -1.39 pp
- Fear: -1.24 pp
- Happiness: -2.33 pp
- Sadness: -0.39 pp
- Surprise: 0.70 pp

## Phi-4-mini-instruct
- Neutral baseline (avg across categories): 28.86%
- Anger: 1.73 pp
- Disgust: 2.00 pp
- Fear: -0.78 pp
- Happiness: 0.28 pp
- Sadness: 0.20 pp
- Surprise: -0.57 pp

## Qwen2.5-0.5B-Instruct
- Neutral baseline (avg across categories): 4.80%
- Anger: -4.51 pp
- Disgust: -4.60 pp
- Fear: -4.80 pp
- Happiness: -4.02 pp
- Sadness: -4.22 pp
- Surprise: -4.18 pp

## Qwen2.5-1.5B-Instruct
- Neutral baseline (avg across categories): 36.44%
- Anger: 0.66 pp
- Disgust: -0.76 pp
- Fear: -0.50 pp
- Happiness: 1.74 pp
- Sadness: 2.08 pp
- Surprise: -2.95 pp

## Qwen2.5-3B-Instruct
- Neutral baseline (avg across categories): 37.58%
- Anger: -8.83 pp
- Disgust: 3.50 pp
- Fear: 1.92 pp
- Happiness: -3.14 pp
- Sadness: -9.99 pp
- Surprise: -0.49 pp

## Qwen3-0.6B
- Neutral baseline (avg across categories): 25.67%
- Anger: -5.63 pp
- Disgust: -6.31 pp
- Fear: -0.81 pp
- Happiness: 0.00 pp
- Sadness: -8.64 pp
- Surprise: -2.86 pp

## Qwen3-1.7B
- Neutral baseline (avg across categories): 29.10%
- Anger: 3.51 pp
- Disgust: -0.06 pp
- Fear: 0.23 pp
- Happiness: 1.51 pp
- Sadness: 1.83 pp
- Surprise: 0.99 pp

## Qwen3-32B-AWQ
- Neutral baseline (avg across categories): 67.28%

## Qwen3-4B
- Neutral baseline (avg across categories): 61.45%
- Anger: 1.69 pp
- Disgust: -1.41 pp
- Fear: 1.02 pp
- Happiness: 1.70 pp
- Sadness: -0.80 pp
- Surprise: 0.37 pp

## Qwen3-8B
- Neutral baseline (avg across categories): 71.17%
- Anger: 0.39 pp
- Disgust: 1.42 pp
- Fear: 0.51 pp
- Happiness: -1.29 pp
- Sadness: -1.16 pp
- Surprise: 1.03 pp

## gemma-3-1b-it
- Neutral baseline (avg across categories): 15.68%
- Anger: 0.08 pp
- Disgust: 0.07 pp
- Fear: 0.00 pp
- Happiness: -0.05 pp
- Sadness: -0.16 pp
- Surprise: -0.11 pp


References (run directories)
- Base: /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live
- Llama-3.2-1B-Instruct | live_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Llama-3.2-1B-Instruct_bfcl_live_multiple_20250927_183933
- Llama-3.2-1B-Instruct | live_parallel | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Llama-3.2-1B-Instruct_bfcl_live_parallel_20250927_225146
- Llama-3.2-1B-Instruct | live_parallel_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Llama-3.2-1B-Instruct_bfcl_live_parallel_multiple_20250927_233538
- Llama-3.2-1B-Instruct | live_simple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Llama-3.2-1B-Instruct_bfcl_live_simple_20250928_022825
- Llama-3.2-3B-Instruct | live_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Llama-3.2-3B-Instruct_bfcl_live_multiple_20250927_191031
- Llama-3.2-3B-Instruct | live_parallel | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Llama-3.2-3B-Instruct_bfcl_live_parallel_20250927_225522
- Llama-3.2-3B-Instruct | live_simple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Llama-3.2-3B-Instruct_bfcl_live_simple_20250928_023959
- Phi-3.5-mini-instruct | live_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Phi-3.5-mini-instruct_bfcl_live_multiple_20250927_193348
- Phi-3.5-mini-instruct | live_parallel | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Phi-3.5-mini-instruct_bfcl_live_parallel_20250927_225922
- Phi-3.5-mini-instruct | live_simple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Phi-3.5-mini-instruct_bfcl_live_simple_20250928_024739
- Phi-4-mini-instruct | live_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Phi-4-mini-instruct_bfcl_live_multiple_20250927_204952
- Phi-4-mini-instruct | live_parallel | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Phi-4-mini-instruct_bfcl_live_parallel_20250927_230437
- Phi-4-mini-instruct | live_parallel_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Phi-4-mini-instruct_bfcl_live_parallel_multiple_20250927_235857
- Phi-4-mini-instruct | live_simple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Phi-4-mini-instruct_bfcl_live_simple_20250928_025806
- Qwen2.5-0.5B-Instruct | live_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Qwen2.5-0.5B-Instruct_bfcl_live_multiple_20250927_163827
- Qwen2.5-0.5B-Instruct | live_parallel | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Qwen2.5-0.5B-Instruct_bfcl_live_parallel_20250927_222411
- Qwen2.5-0.5B-Instruct | live_parallel_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Qwen2.5-0.5B-Instruct_bfcl_live_parallel_multiple_20250927_231438
- Qwen2.5-0.5B-Instruct | live_simple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Qwen2.5-0.5B-Instruct_bfcl_live_simple_20250928_014158
- Qwen2.5-1.5B-Instruct | live_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Qwen2.5-1.5B-Instruct_bfcl_live_multiple_20250927_170318
- Qwen2.5-1.5B-Instruct | live_parallel | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Qwen2.5-1.5B-Instruct_bfcl_live_parallel_20250927_222749
- Qwen2.5-1.5B-Instruct | live_parallel_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Qwen2.5-1.5B-Instruct_bfcl_live_parallel_multiple_20250927_231827
- Qwen2.5-1.5B-Instruct | live_simple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Qwen2.5-1.5B-Instruct_bfcl_live_simple_20250928_015012
- Qwen2.5-3B-Instruct | live_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Qwen2.5-3B-Instruct_bfcl_live_multiple_20250927_171605
- Qwen2.5-3B-Instruct | live_parallel | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Qwen2.5-3B-Instruct_bfcl_live_parallel_20250927_223007
- Qwen2.5-3B-Instruct | live_parallel_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Qwen2.5-3B-Instruct_bfcl_live_parallel_multiple_20250927_232034
- Qwen2.5-3B-Instruct | live_simple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Qwen2.5-3B-Instruct_bfcl_live_simple_20250928_015717
- Qwen3-0.6B | live_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Qwen3-0.6B_bfcl_live_multiple_20250927_173130
- Qwen3-0.6B | live_parallel | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Qwen3-0.6B_bfcl_live_parallel_20250927_223353
- Qwen3-0.6B | live_parallel_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Qwen3-0.6B_bfcl_live_parallel_multiple_20250927_232318
- Qwen3-0.6B | live_simple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Qwen3-0.6B_bfcl_live_simple_20250928_020225
- Qwen3-1.7B | live_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Qwen3-1.7B_bfcl_live_multiple_20250927_175148
- Qwen3-1.7B | live_parallel | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Qwen3-1.7B_bfcl_live_parallel_20250927_223958
- Qwen3-1.7B | live_parallel_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Qwen3-1.7B_bfcl_live_parallel_multiple_20250927_232821
- Qwen3-1.7B | live_simple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Qwen3-1.7B_bfcl_live_simple_20250928_020940
- Qwen3-32B-AWQ | live_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Qwen3-32B-AWQ_bfcl_live_multiple_20250929_022046
- Qwen3-32B-AWQ | live_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Qwen3-32B-AWQ_bfcl_live_multiple_20250929_023139
- Qwen3-32B-AWQ | live_parallel | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Qwen3-32B-AWQ_bfcl_live_parallel_20250929_023557
- Qwen3-32B-AWQ | live_parallel_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Qwen3-32B-AWQ_bfcl_live_parallel_multiple_20250929_023821
- Qwen3-32B-AWQ | live_simple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Qwen3-32B-AWQ_bfcl_live_simple_20250929_024031
- Qwen3-4B | live_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Qwen3-4B_bfcl_live_multiple_20250927_181341
- Qwen3-4B | live_parallel | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Qwen3-4B_bfcl_live_parallel_20250927_224519
- Qwen3-4B | live_parallel_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Qwen3-4B_bfcl_live_parallel_multiple_20250927_233117
- Qwen3-4B | live_simple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Qwen3-4B_bfcl_live_simple_20250928_021346
- Qwen3-8B | live_simple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/Qwen3-8B_bfcl_live_simple_20250928_022005
- gemma-3-1b-it | live_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/gemma-3-1b-it_bfcl_live_multiple_20250927_215130
- gemma-3-1b-it | live_parallel | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/gemma-3-1b-it_bfcl_live_parallel_20250927_230947
- gemma-3-1b-it | live_parallel_multiple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/gemma-3-1b-it_bfcl_live_parallel_multiple_20250928_010240
- gemma-3-1b-it | live_simple | /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/results/bfcl/live/gemma-3-1b-it_bfcl_live_simple_20250928_030328