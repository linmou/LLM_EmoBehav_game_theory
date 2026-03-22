# Game-Theory Decision Impact Report (vs neutral)

## Data Used
- Root scanned: `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games`
- Input files searched: `**/summary_choice_ratio.csv`, `**/summary_behavior_ratio.csv`
- Latest run per (model, game_setting): 52
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Llama-3.2-1B-Instruct_game_theory_Trust_Game_Trustee_20251229_083150`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Llama-3.2-1B-Instruct_game_theory_Trust_Game_Trustor_20251229_071655`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Llama-3.2-1B-Instruct_game_theory_Ultimatum_Game_Proposer_20251229_094052`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Llama-3.2-1B-Instruct_game_theory_Ultimatum_Game_Responder_20251229_104704`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Llama-3.2-3B-Instruct_game_theory_Trust_Game_Trustee_20251229_083748`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Llama-3.2-3B-Instruct_game_theory_Trust_Game_Trustor_20251229_072305`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Llama-3.2-3B-Instruct_game_theory_Ultimatum_Game_Proposer_20251229_094636`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Llama-3.2-3B-Instruct_game_theory_Ultimatum_Game_Responder_20251229_105216`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Phi-3.5-mini-instruct_game_theory_Trust_Game_Trustee_20251229_081611`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Phi-3.5-mini-instruct_game_theory_Trust_Game_Trustor_20251229_065927`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Phi-3.5-mini-instruct_game_theory_Ultimatum_Game_Proposer_20251229_092628`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Phi-3.5-mini-instruct_game_theory_Ultimatum_Game_Responder_20251229_103305`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Phi-4-mini-instruct_game_theory_Trust_Game_Trustee_20251229_082725`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Phi-4-mini-instruct_game_theory_Trust_Game_Trustor_20251229_071102`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Phi-4-mini-instruct_game_theory_Ultimatum_Game_Proposer_20251229_093629`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Phi-4-mini-instruct_game_theory_Ultimatum_Game_Responder_20251229_104255`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Qwen2.5-0.5B-Instruct_game_theory_Trust_Game_Trustee_20251229_084529`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Qwen2.5-0.5B-Instruct_game_theory_Trust_Game_Trustor_20251229_073353`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Qwen2.5-0.5B-Instruct_game_theory_Ultimatum_Game_Proposer_20251229_095431`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Qwen2.5-0.5B-Instruct_game_theory_Ultimatum_Game_Responder_20251229_105932`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Qwen2.5-1.5B-Instruct_game_theory_Trust_Game_Trustee_20251229_085006`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Qwen2.5-1.5B-Instruct_game_theory_Trust_Game_Trustor_20251229_073906`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Qwen2.5-1.5B-Instruct_game_theory_Ultimatum_Game_Proposer_20251229_095907`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Qwen2.5-1.5B-Instruct_game_theory_Ultimatum_Game_Responder_20251229_110340`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Qwen2.5-3B-Instruct_game_theory_Trust_Game_Trustee_20251229_085338`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Qwen2.5-3B-Instruct_game_theory_Trust_Game_Trustor_20251229_074226`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Qwen2.5-3B-Instruct_game_theory_Ultimatum_Game_Proposer_20251229_100242`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Qwen2.5-3B-Instruct_game_theory_Ultimatum_Game_Responder_20251229_110632`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Qwen3-0.6B_game_theory_Trust_Game_Trustee_20251229_085727`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Qwen3-0.6B_game_theory_Trust_Game_Trustor_20251229_074624`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Qwen3-0.6B_game_theory_Ultimatum_Game_Proposer_20251229_100629`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Qwen3-0.6B_game_theory_Ultimatum_Game_Responder_20251229_110954`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Qwen3-1.7B_game_theory_Trust_Game_Trustee_20251229_090022`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Qwen3-1.7B_game_theory_Trust_Game_Trustor_20251229_074907`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Qwen3-1.7B_game_theory_Ultimatum_Game_Proposer_20251229_101004`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Qwen3-1.7B_game_theory_Ultimatum_Game_Responder_20251229_111238`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Qwen3-4B_game_theory_Trust_Game_Trustee_20251229_090342`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Qwen3-4B_game_theory_Trust_Game_Trustor_20251229_075302`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Qwen3-4B_game_theory_Ultimatum_Game_Proposer_20251229_101331`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/Qwen3-4B_game_theory_Ultimatum_Game_Responder_20251229_111531`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/gemma-3-1b-it_game_theory_Trust_Game_Trustee_20251229_080106`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/gemma-3-1b-it_game_theory_Trust_Game_Trustor_20251229_064009`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/gemma-3-1b-it_game_theory_Ultimatum_Game_Proposer_20251229_091040`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/gemma-3-1b-it_game_theory_Ultimatum_Game_Responder_20251229_102006`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/gemma-3-270m-it_game_theory_Trust_Game_Trustee_20251229_075820`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/gemma-3-270m-it_game_theory_Trust_Game_Trustor_20251229_063426`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/gemma-3-270m-it_game_theory_Ultimatum_Game_Proposer_20251229_090710`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/gemma-3-270m-it_game_theory_Ultimatum_Game_Responder_20251229_101700`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/gemma-3-4b-it_game_theory_Trust_Game_Trustee_20251229_080545`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/gemma-3-4b-it_game_theory_Trust_Game_Trustor_20251229_064504`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/gemma-3-4b-it_game_theory_Ultimatum_Game_Proposer_20251229_091601`
  - `results/new_game_theory/shuffle_crowd-enVent_textlike_sequential_games/gemma-3-4b-it_game_theory_Ultimatum_Game_Responder_20251229_102500`

## Method
- For each `(model, game_setting)`, select the latest timestamped run directory.
- Collapse intensity by averaging `ratio` over all intensities present.
- Compute `delta_vs_neutral = ratio(emotion) - ratio(neutral)` for each option/behavior.
- Summarize best/worst emotion deltas and `delta_range = best - worst`.
- Additionally, compute per-emotion `delta_range_across_intensity` with per-intensity deltas (vs neutral mean).
- In per-game tables, show deltas for all emotions (vs neutral), ranked by Δ descending.
- When `raw_results.json` is available, annotate each emotion as `emo:Δ{sig}[ci_low,ci_high]`.
  - `{sig}` is `! / !! / !!!` based on Benjamini–Hochberg FDR per game-setting (within Option table / Behavior table).

## Outputs
- Option CSV: `test_output/tmp_game_theory_report/option_impacted_by_emo_vs_neutral_latest.csv`
- Behavior CSV: `test_output/tmp_game_theory_report/behavior_impacted_emo_vs_neutral_latest.csv`
- Option-by-intensity CSV: `test_output/tmp_game_theory_report/option_intensity_impacted_by_emo_vs_neutral_latest.csv`
- Behavior-by-intensity CSV: `test_output/tmp_game_theory_report/behavior_intensity_impacted_emo_vs_neutral_latest.csv`

## Strongest Option Effects (Top 20 by delta_range)
| game_setting | model | option_id | neutral | best (Δ) | worst (Δ) | range |
|---|---|---|---:|---|---|---:|
| Ultimatum_Game_Responder | Phi-4-mini-instruct | -1 | 0.099 | sadness (+0.547) | happiness (-0.092) | 0.640 |
| Trust_Game_Trustee | Llama-3.2-1B-Instruct | -1 | 0.095 | sadness (+0.713) | fear (+0.080) | 0.633 |
| Trust_Game_Trustee | Phi-4-mini-instruct | -1 | 0.014 | sadness (+0.618) | happiness (-0.001) | 0.619 |
| Ultimatum_Game_Proposer | Phi-4-mini-instruct | -1 | 0.000 | sadness (+0.535) | fear (+0.000) | 0.535 |
| Trust_Game_Trustor | Llama-3.2-1B-Instruct | -1 | 0.152 | sadness (+0.732) | disgust (+0.229) | 0.503 |
| Ultimatum_Game_Proposer | Qwen2.5-0.5B-Instruct | 1 | 0.540 | happiness (+0.098) | anger (-0.340) | 0.438 |
| Trust_Game_Trustor | Phi-4-mini-instruct | -1 | 0.001 | sadness (+0.416) | happiness (+0.001) | 0.414 |
| Ultimatum_Game_Responder | Llama-3.2-1B-Instruct | -1 | 0.007 | sadness (+0.402) | fear (-0.002) | 0.404 |
| Ultimatum_Game_Responder | Phi-4-mini-instruct | 2 | 0.480 | surprise (+0.048) | sadness (-0.337) | 0.384 |
| Trust_Game_Trustee | Qwen2.5-0.5B-Instruct | 1 | 0.455 | happiness (+0.119) | anger (-0.263) | 0.382 |
| Ultimatum_Game_Proposer | Llama-3.2-1B-Instruct | -1 | 0.010 | sadness (+0.443) | happiness (+0.074) | 0.370 |
| Ultimatum_Game_Responder | Llama-3.2-1B-Instruct | 2 | 0.346 | fear (+0.190) | sadness (-0.175) | 0.365 |
| Trust_Game_Trustor | Llama-3.2-3B-Instruct | -1 | 0.074 | surprise (+0.330) | disgust (-0.003) | 0.333 |
| Ultimatum_Game_Proposer | Qwen2.5-3B-Instruct | 1 | 0.466 | happiness (+0.171) | anger (-0.129) | 0.300 |
| Ultimatum_Game_Proposer | Qwen2.5-0.5B-Instruct | 3 | 0.245 | anger (+0.225) | happiness (-0.066) | 0.290 |
| Trust_Game_Trustor | Qwen2.5-0.5B-Instruct | -1 | 0.174 | happiness (+0.164) | fear (-0.119) | 0.282 |
| Trust_Game_Trustee | Llama-3.2-3B-Instruct | -1 | 0.086 | surprise (+0.256) | fear (-0.025) | 0.280 |
| Trust_Game_Trustee | Qwen2.5-0.5B-Instruct | 3 | 0.249 | anger (+0.209) | disgust (-0.065) | 0.274 |
| Ultimatum_Game_Proposer | Llama-3.2-3B-Instruct | 1 | 0.491 | sadness (+0.164) | disgust (-0.110) | 0.274 |
| Trust_Game_Trustee | Qwen2.5-0.5B-Instruct | 2 | 0.264 | fear (+0.098) | sadness (-0.174) | 0.272 |

## Per Game Setting Option (All models)
### Trust_Game_Trustee
| model | option_id | neutral | all emotion deltas (Δ vs neutral) | range |
|---|---|---:|---|---:|
| Llama-3.2-1B-Instruct | -1 | 0.095 | sadness:+0.713333;surprise:+0.662083;happiness:+0.356250;anger:+0.302500;disgust:+0.170000;fear:+0.080000 | 0.633 |
| Llama-3.2-1B-Instruct | 1 | 0.599 | anger:-0.222500;fear:-0.250417;happiness:-0.255833;disgust:-0.258333;sadness:-0.425000;surprise:-0.451667 | 0.229 |
| Llama-3.2-1B-Instruct | 2 | 0.214 | disgust:+0.072083;fear:+0.070417;happiness:-0.050000;anger:-0.055000;surprise:-0.119375;sadness:-0.138750 | 0.211 |
| Llama-3.2-1B-Instruct | 3 | 0.092 | fear:+0.100000;disgust:+0.016250;anger:-0.025000;surprise:-0.043125;happiness:-0.050417;sadness:-0.053750 | 0.154 |
| Llama-3.2-3B-Instruct | -1 | 0.086 | surprise:+0.255833;sadness:+0.249583;happiness:+0.027500;anger:+0.000417;disgust:+0.000000;fear:-0.024583 | 0.280 |
| Llama-3.2-3B-Instruct | 1 | 0.400 | surprise:+0.000[+0.000,+0.000]; anger:-0.500[-1.000,+0.000]; fear:-0.500[-1.000,+0.000]; sadness:-1.000[-1.000,-1.000] | 0.118 |
| Llama-3.2-3B-Instruct | 2 | 0.282 | sadness:+1.000[+1.000,+1.000]; fear:+0.500[+0.000,+1.000]; anger:+0.000[+0.000,+0.000]; surprise:+0.000[+0.000,+0.000] | 0.118 |
| Llama-3.2-3B-Instruct | 3 | 0.231 | anger:+0.500[+0.000,+1.000]; fear:+0.000[-1.000,+1.000]; sadness:+0.000[+0.000,+0.000]; surprise:+0.000[+0.000,+0.000] | 0.140 |
| Phi-3.5-mini-instruct | -1 | 0.010 | disgust:+0.003750;sadness:+0.002500;anger:-0.001667;fear:-0.001667;surprise:-0.005000;happiness:-0.007083 | 0.011 |
| Phi-3.5-mini-instruct | 1 | 0.431 | surprise:+0.026[-0.026,+0.077]; happiness:+0.018[-0.042,+0.076]; sadness:-0.021[-0.072,+0.034]; disgust:-0.044[-0.101,+0.012]; anger:-0.056[-0.114,+0.002]; fear:-0.089[-0.151,-0.031] | 0.075 |
| Phi-3.5-mini-instruct | 2 | 0.295 | fear:+0.050[-0.010,+0.102]; disgust:+0.021[-0.028,+0.074]; surprise:+0.011[-0.037,+0.062]; anger:-0.004[-0.054,+0.045]; happiness:-0.005[-0.065,+0.053]; sadness:-0.025[-0.077,+0.028] | 0.029 |
| Phi-3.5-mini-instruct | 3 | 0.264 | anger:+0.059[+0.005,+0.108]; sadness:+0.046[-0.010,+0.093]; fear:+0.039[-0.015,+0.093]; disgust:+0.023[-0.028,+0.073]; happiness:-0.012[-0.065,+0.039]; surprise:-0.037[-0.084,+0.013] | 0.042 |
| Phi-4-mini-instruct | -1 | 0.014 | sadness:+0.617917;disgust:+0.121667;anger:+0.059583;fear:+0.018750;surprise:+0.008333;happiness:-0.000833 | 0.619 |
| Phi-4-mini-instruct | 1 | 0.390 | happiness:+0.667[+0.000,+1.000]; disgust:+0.000[+0.000,+0.000] | 0.212 |
| Phi-4-mini-instruct | 2 | 0.300 | disgust:+0.000[+0.000,+0.000]; happiness:-0.333[-0.667,+0.000] | 0.222 |
| Phi-4-mini-instruct | 3 | 0.296 | disgust:+0.000[+0.000,+0.000]; happiness:-0.333[-1.000,+0.000] | 0.215 |
| Qwen2.5-0.5B-Instruct | -1 | 0.033 | sadness:+0.129167;disgust:+0.081667;happiness:+0.055000;surprise:+0.042917;anger:+0.041667;fear:-0.003750 | 0.133 |
| Qwen2.5-0.5B-Instruct | 1 | 0.455 | happiness:+0.103![+0.032,+0.173]; disgust:+0.073[+0.016,+0.136]; sadness:+0.016[-0.068,+0.096]; surprise:-0.049[-0.114,+0.011]; anger:-0.122![-0.202,-0.038]; fear:-0.151!!![-0.210,-0.090] | 0.382 |
| Qwen2.5-0.5B-Instruct | 2 | 0.264 | fear:+0.118!!![+0.059,+0.171]; anger:+0.063[-0.008,+0.143]; surprise:+0.011[-0.057,+0.079]; disgust:-0.039[-0.094,+0.013]; happiness:-0.106!!![-0.158,-0.053]; sadness:-0.131!![-0.191,-0.068] | 0.272 |
| Qwen2.5-0.5B-Instruct | 3 | 0.249 | sadness:+0.116![+0.044,+0.191]; anger:+0.059[-0.017,+0.130]; surprise:+0.038[-0.019,+0.095]; fear:+0.033[-0.022,+0.083]; happiness:+0.003[-0.062,+0.065]; disgust:-0.034[-0.084,+0.018] | 0.274 |
| Qwen2.5-1.5B-Instruct | -1 | 0.011 | fear:+0.012083;disgust:+0.005417;anger:+0.002500;happiness:+0.002083;sadness:+0.002083;surprise:-0.000833 | 0.013 |
| Qwen2.5-1.5B-Instruct | 1 | 0.554 | happiness:+0.196[+0.039,+0.333]; disgust:+0.143[-0.429,+0.571]; surprise:+0.136[-0.023,+0.273]; fear:+0.044[-0.133,+0.244]; anger:-0.071[-0.214,+0.089]; sadness:-0.086[-0.214,+0.043] | 0.087 |
| Qwen2.5-1.5B-Instruct | 2 | 0.217 | disgust:+0.143[+0.000,+0.429]; sadness:+0.057[-0.071,+0.186]; fear:+0.044[-0.111,+0.200]; anger:-0.018[-0.161,+0.107]; happiness:-0.059[-0.176,+0.059]; surprise:-0.091[-0.182,+0.000] | 0.020 |
| Qwen2.5-1.5B-Instruct | 3 | 0.217 | anger:+0.089[-0.036,+0.214]; sadness:+0.029[-0.086,+0.129]; surprise:-0.045[-0.182,+0.091]; fear:-0.089[-0.244,+0.044]; happiness:-0.137[-0.275,-0.020]; disgust:-0.286[-0.714,+0.286] | 0.061 |
| Qwen2.5-3B-Instruct | -1 | 0.004 | sadness:-0.000833;fear:-0.001667;happiness:-0.001667;disgust:-0.002083;anger:-0.002500;surprise:-0.002500 | 0.002 |
| Qwen2.5-3B-Instruct | 1 | 0.650 | disgust:+0.130!!![+0.087,+0.168]; happiness:+0.071!![+0.032,+0.114]; surprise:-0.011[-0.057,+0.032]; sadness:-0.025[-0.070,+0.015]; anger:-0.055[-0.099,-0.011]; fear:-0.066![-0.117,-0.018] | 0.162 |
| Qwen2.5-3B-Instruct | 2 | 0.168 | fear:+0.067!![+0.023,+0.104]; anger:+0.044[+0.001,+0.083]; sadness:+0.039[+0.004,+0.072]; surprise:+0.025[-0.013,+0.065]; happiness:-0.028[-0.061,+0.008]; disgust:-0.053![-0.088,-0.018] | 0.122 |
| Qwen2.5-3B-Instruct | 3 | 0.179 | anger:+0.010[-0.024,+0.051]; fear:-0.001[-0.039,+0.038]; surprise:-0.014[-0.052,+0.023]; sadness:-0.014[-0.051,+0.022]; happiness:-0.043![-0.074,-0.009]; disgust:-0.076!!![-0.107,-0.042] | 0.048 |
| Qwen3-0.6B | -1 | 0.000 | anger:+0.006250;fear:+0.006250;disgust:+0.005417;happiness:+0.002500;surprise:+0.001667;sadness:+0.000000 | 0.006 |
| Qwen3-0.6B | 1 | 0.441 | anger:+0.086[-0.025,+0.191]; surprise:+0.073[-0.006,+0.147]; fear:+0.059[-0.034,+0.151]; sadness:+0.057[+0.000,+0.116]; happiness:+0.015[-0.042,+0.073]; disgust:-0.049[-0.120,+0.019] | 0.120 |
| Qwen3-0.6B | 2 | 0.286 | disgust:+0.062[-0.005,+0.125]; happiness:+0.015[-0.035,+0.081]; anger:-0.012[-0.105,+0.086]; sadness:-0.017[-0.074,+0.037]; fear:-0.025[-0.101,+0.042]; surprise:-0.067[-0.144,+0.003] | 0.060 |
| Qwen3-0.6B | 3 | 0.273 | surprise:-0.006[-0.073,+0.064]; disgust:-0.014[-0.082,+0.043]; happiness:-0.029[-0.081,+0.018]; fear:-0.034[-0.113,+0.050]; sadness:-0.040[-0.099,+0.025]; anger:-0.074[-0.173,+0.019] | 0.071 |
| Qwen3-1.7B | -1 | 0.001 | disgust:+0.000000;happiness:+0.000000;sadness:+0.000000;surprise:+0.000000;anger:-0.001250;fear:-0.001250 | 0.001 |
| Qwen3-1.7B | 1 | 0.541 | surprise:+0.000[-0.079,+0.071]; disgust:-0.004[-0.097,+0.073]; happiness:-0.025[-0.104,+0.050]; sadness:-0.026[-0.111,+0.048]; anger:-0.040[-0.117,+0.040]; fear:-0.072[-0.158,+0.009] | 0.043 |
| Qwen3-1.7B | 2 | 0.270 | anger:-0.018[-0.099,+0.047]; sadness:-0.018[-0.100,+0.059]; surprise:-0.019[-0.090,+0.056]; happiness:-0.046[-0.120,+0.025]; disgust:-0.065[-0.137,+0.012]; fear:-0.068[-0.145,+0.014] | 0.040 |
| Qwen3-1.7B | 3 | 0.188 | fear:+0.140![+0.063,+0.217]; happiness:+0.071[-0.004,+0.154]; disgust:+0.069[-0.008,+0.137]; anger:+0.058[-0.011,+0.128]; sadness:+0.044[-0.026,+0.118]; surprise:+0.019[-0.045,+0.079] | 0.036 |
| Qwen3-4B | 1 | 0.574 | happiness:+0.014[-0.036,+0.060]; anger:+0.013[-0.033,+0.060]; sadness:+0.011[-0.035,+0.059]; surprise:-0.003[-0.050,+0.048]; fear:-0.015[-0.066,+0.031]; disgust:-0.025[-0.071,+0.021] | 0.043 |
| Qwen3-4B | 2 | 0.255 | disgust:+0.025[-0.015,+0.068]; happiness:+0.024[-0.020,+0.063]; fear:-0.001[-0.045,+0.035]; anger:-0.003[-0.044,+0.038]; sadness:-0.006[-0.046,+0.035]; surprise:-0.011[-0.051,+0.025] | 0.033 |
| Qwen3-4B | 3 | 0.171 | fear:+0.016[-0.016,+0.050]; surprise:+0.014[-0.021,+0.048]; disgust:+0.000[-0.035,+0.035]; sadness:-0.005[-0.043,+0.030]; anger:-0.010[-0.041,+0.021]; happiness:-0.038[-0.073,+0.003] | 0.053 |
| gemma-3-1b-it | -1 | 0.043 | happiness:+0.008750;fear:+0.008333;sadness:+0.007917;disgust:+0.006250;surprise:+0.005000;anger:+0.004167 | 0.005 |
| gemma-3-1b-it | 1 | 0.511 | anger:-0.014583;disgust:-0.019167;fear:-0.035000;surprise:-0.035417;happiness:-0.037917;sadness:-0.041667 | 0.027 |
| gemma-3-1b-it | 2 | 0.280 | sadness:+0.045833;happiness:+0.045000;anger:+0.043750;surprise:+0.037917;fear:+0.025417;disgust:+0.017083 | 0.029 |
| gemma-3-1b-it | 3 | 0.166 | fear:+0.001250;disgust:-0.004167;surprise:-0.007500;sadness:-0.012083;happiness:-0.015833;anger:-0.033333 | 0.035 |
| gemma-3-270m-it | -1 | 0.399 | disgust:+0.013750;happiness:+0.012083;anger:+0.009583;surprise:+0.007917;sadness:+0.005000;fear:+0.000833 | 0.013 |
| gemma-3-270m-it | 1 | 0.131 | happiness:+0.013333;fear:+0.005000;anger:+0.002500;disgust:+0.000833;surprise:+0.000000;sadness:-0.003750 | 0.017 |
| gemma-3-270m-it | 2 | 0.360 | fear:-0.005000;anger:-0.005833;sadness:-0.006250;surprise:-0.006250;happiness:-0.011250;disgust:-0.019167 | 0.014 |
| gemma-3-270m-it | 3 | 0.110 | sadness:+0.005000;disgust:+0.004583;fear:-0.000833;surprise:-0.001667;anger:-0.006250;happiness:-0.014167 | 0.019 |
| gemma-3-4b-it | -1 | 0.095 | surprise:+0.000000;fear:-0.001250;sadness:-0.001667;anger:-0.002083;disgust:-0.002083;happiness:-0.004167 | 0.004 |
| gemma-3-4b-it | 1 | 0.517 | happiness:+0.040417;sadness:+0.021250;surprise:+0.009583;anger:+0.007500;disgust:+0.001250;fear:-0.010000 | 0.050 |
| gemma-3-4b-it | 2 | 0.254 | disgust:-0.015000;fear:-0.016667;anger:-0.024583;surprise:-0.029167;sadness:-0.031250;happiness:-0.052917 | 0.038 |
| gemma-3-4b-it | 3 | 0.134 | fear:+0.027917;surprise:+0.019583;anger:+0.019167;happiness:+0.016667;disgust:+0.015833;sadness:+0.011667 | 0.016 |

### Trust_Game_Trustor
| model | option_id | neutral | all emotion deltas (Δ vs neutral) | range |
|---|---|---:|---|---:|
| Llama-3.2-1B-Instruct | -1 | 0.152 | sadness:+0.732083;surprise:+0.589583;happiness:+0.373750;anger:+0.289583;fear:+0.236667;disgust:+0.228750 | 0.503 |
| Llama-3.2-1B-Instruct | 1 | 0.562 | anger:-0.218750;happiness:-0.266250;fear:-0.276667;disgust:-0.278750;surprise:-0.353125;sadness:-0.449375 | 0.231 |
| Llama-3.2-1B-Instruct | 2 | 0.189 | disgust:+0.008750;fear:-0.010000;anger:-0.048333;happiness:-0.062500;surprise:-0.116250;sadness:-0.133750 | 0.142 |
| Llama-3.2-1B-Instruct | 3 | 0.096 | fear:+0.050000;disgust:+0.041250;anger:-0.022500;surprise:-0.027500;happiness:-0.045000;sadness:-0.063750 | 0.114 |
| Llama-3.2-3B-Instruct | -1 | 0.074 | surprise:+0.330417;sadness:+0.303333;fear:+0.117500;anger:+0.033333;happiness:+0.006667;disgust:-0.002500 | 0.333 |
| Llama-3.2-3B-Instruct | 1 | 0.412 | fear:+0.000[-1.000,+1.000] | 0.146 |
| Llama-3.2-3B-Instruct | 2 | 0.276 | fear:+0.333[+0.000,+1.000] | 0.129 |
| Llama-3.2-3B-Instruct | 3 | 0.237 | fear:-0.333[-1.000,+0.000] | 0.142 |
| Phi-3.5-mini-instruct | -1 | 0.001 | surprise:+0.003333;sadness:+0.002917;disgust:+0.002083;anger:+0.001667;fear:+0.001250;happiness:+0.000417 | 0.003 |
| Phi-3.5-mini-instruct | 1 | 0.476 | happiness:+0.055[+0.011,+0.104]; surprise:+0.027[-0.031,+0.075]; sadness:+0.013[-0.036,+0.066]; anger:-0.005[-0.061,+0.041]; disgust:-0.012[-0.060,+0.033]; fear:-0.012[-0.063,+0.039] | 0.063 |
| Phi-3.5-mini-instruct | 2 | 0.249 | surprise:+0.039[-0.011,+0.081]; fear:+0.032[-0.013,+0.080]; sadness:+0.019[-0.027,+0.060]; anger:+0.017[-0.029,+0.064]; disgust:+0.016[-0.031,+0.064]; happiness:-0.025[-0.067,+0.015] | 0.056 |
| Phi-3.5-mini-instruct | 3 | 0.274 | disgust:-0.004[-0.055,+0.040]; anger:-0.012[-0.054,+0.035]; fear:-0.020[-0.066,+0.027]; happiness:-0.029[-0.072,+0.011]; sadness:-0.032[-0.076,+0.007]; surprise:-0.065[-0.103,-0.024] | 0.052 |
| Phi-4-mini-instruct | -1 | 0.001 | sadness:+0.415625;disgust:+0.020417;anger:+0.019167;surprise:+0.006250;fear:+0.002500;happiness:+0.001250 | 0.414 |
| Phi-4-mini-instruct | 1 | 0.472 | fear:+0.075[+0.014,+0.127]; sadness:+0.024[-0.035,+0.085]; happiness:-0.009[-0.062,+0.045]; disgust:-0.028[-0.088,+0.030]; surprise:-0.115[-0.264,+0.046]; anger:-0.139[-0.245,-0.040] | 0.177 |
| Phi-4-mini-instruct | 2 | 0.287 | surprise:+0.126[-0.034,+0.276]; anger:+0.119[+0.007,+0.238]; disgust:-0.011[-0.077,+0.047]; sadness:-0.036[-0.092,+0.017]; fear:-0.036[-0.094,+0.016]; happiness:-0.047[-0.116,+0.006] | 0.089 |
| Phi-4-mini-instruct | 3 | 0.239 | happiness:+0.056[+0.004,+0.109]; disgust:+0.039[-0.017,+0.096]; anger:+0.020[-0.093,+0.126]; sadness:+0.012[-0.043,+0.059]; surprise:-0.011[-0.126,+0.103]; fear:-0.038[-0.085,+0.014] | 0.088 |
| Qwen2.5-0.5B-Instruct | -1 | 0.174 | happiness:+0.163750;surprise:+0.015833;disgust:+0.001250;sadness:-0.019583;anger:-0.074167;fear:-0.118750 | 0.282 |
| Qwen2.5-0.5B-Instruct | 1 | 0.246 | sadness:+0.254!!![+0.182,+0.315]; happiness:+0.125!![+0.046,+0.198]; fear:+0.031[-0.056,+0.113]; disgust:+0.021[-0.042,+0.087]; surprise:-0.008[-0.068,+0.045]; anger:-0.051[-0.141,+0.038] | 0.237 |
| Qwen2.5-0.5B-Instruct | 2 | 0.344 | fear:+0.097[+0.000,+0.179]; anger:+0.090[-0.064,+0.231]; surprise:+0.037[-0.037,+0.105]; disgust:+0.004[-0.083,+0.092]; happiness:-0.095![-0.165,-0.024]; sadness:-0.192!!![-0.246,-0.131] | 0.271 |
| Qwen2.5-0.5B-Instruct | 3 | 0.236 | disgust:-0.025[-0.104,+0.062]; surprise:-0.028[-0.096,+0.034]; happiness:-0.030[-0.095,+0.030]; anger:-0.038[-0.167,+0.103]; sadness:-0.062[-0.123,-0.005]; fear:-0.128![-0.210,-0.041] | 0.128 |
| Qwen2.5-1.5B-Instruct | -1 | 0.014 | fear:+0.006250;sadness:+0.004583;anger:+0.002917;surprise:+0.002917;happiness:-0.000417;disgust:-0.003333 | 0.010 |
| Qwen2.5-1.5B-Instruct | 1 | 0.646 | happiness:+0.056[+0.005,+0.098]; surprise:-0.008[-0.051,+0.041]; disgust:-0.045[-0.099,+0.004]; fear:-0.056[-0.109,-0.005]; sadness:-0.059[-0.108,-0.017]; anger:-0.090!![-0.141,-0.042] | 0.115 |
| Qwen2.5-1.5B-Instruct | 2 | 0.181 | disgust:+0.041[-0.004,+0.090]; anger:+0.032[-0.009,+0.073]; sadness:+0.025[-0.014,+0.062]; surprise:+0.024[-0.014,+0.061]; fear:+0.020[-0.021,+0.060]; happiness:+0.004[-0.037,+0.045] | 0.029 |
| Qwen2.5-1.5B-Instruct | 3 | 0.159 | anger:+0.058![+0.015,+0.099]; fear:+0.036[-0.003,+0.076]; sadness:+0.034[+0.000,+0.071]; disgust:+0.004[-0.038,+0.047]; surprise:-0.016[-0.053,+0.021]; happiness:-0.060!![-0.096,-0.021] | 0.092 |
| Qwen2.5-3B-Instruct | -1 | 0.003 | anger:+0.001875;happiness:+0.001250;fear:+0.000833;disgust:-0.000417;sadness:-0.000417;surprise:-0.000417 | 0.002 |
| Qwen2.5-3B-Instruct | 1 | 0.661 | disgust:+0.156!!![+0.117,+0.193]; happiness:+0.081!![+0.035,+0.122]; sadness:-0.026[-0.073,+0.021]; surprise:-0.045[-0.093,-0.001]; anger:-0.073!![-0.119,-0.028]; fear:-0.094!!![-0.142,-0.048] | 0.183 |
| Qwen2.5-3B-Instruct | 2 | 0.149 | fear:+0.057!![+0.020,+0.093]; anger:+0.036[-0.001,+0.073]; surprise:+0.029[-0.003,+0.062]; sadness:+0.019[-0.014,+0.055]; happiness:-0.031[-0.068,+0.004]; disgust:-0.065!!![-0.096,-0.035] | 0.107 |
| Qwen2.5-3B-Instruct | 3 | 0.188 | fear:+0.038[+0.001,+0.078]; anger:+0.036[-0.004,+0.072]; surprise:+0.016[-0.020,+0.055]; sadness:+0.008[-0.031,+0.044]; happiness:-0.049![-0.081,-0.014]; disgust:-0.091!!![-0.121,-0.060] | 0.075 |
| Qwen3-0.6B | -1 | 0.001 | fear:+0.005000;happiness:+0.002500;disgust:+0.000833;surprise:+0.000000;anger:-0.001250;sadness:-0.001250 | 0.006 |
| Qwen3-0.6B | 1 | 0.519 | anger:+0.280[+0.080,+0.440]; sadness:+0.103[+0.028,+0.176]; surprise:+0.072[-0.015,+0.165]; fear:+0.051[-0.027,+0.125]; happiness:-0.019[-0.074,+0.037]; disgust:-0.089[-0.152,-0.037] | 0.122 |
| Qwen3-0.6B | 2 | 0.300 | disgust:+0.035[-0.033,+0.089]; happiness:+0.017[-0.033,+0.068]; surprise:-0.026[-0.108,+0.062]; fear:-0.047[-0.118,+0.027]; sadness:-0.079[-0.145,-0.007]; anger:-0.220[-0.360,-0.060] | 0.051 |
| Qwen3-0.6B | 3 | 0.180 | disgust:+0.055[+0.014,+0.102]; happiness:+0.002[-0.043,+0.048]; fear:-0.004[-0.059,+0.055]; sadness:-0.024[-0.079,+0.038]; surprise:-0.046[-0.108,+0.026]; anger:-0.060[-0.180,+0.060] | 0.091 |
| Qwen3-1.7B | 1 | 0.575 | anger:+0.000[-0.048,+0.051]; sadness:+0.000[-0.051,+0.045]; disgust:-0.004[-0.056,+0.041]; happiness:-0.013[-0.067,+0.037]; surprise:-0.017[-0.068,+0.032]; fear:-0.049[-0.097,-0.001] | 0.051 |
| Qwen3-1.7B | 2 | 0.271 | happiness:+0.023[-0.021,+0.067]; fear:+0.019[-0.030,+0.069]; anger:-0.003[-0.054,+0.038]; disgust:-0.010[-0.054,+0.038]; surprise:-0.019[-0.063,+0.023]; sadness:-0.020[-0.064,+0.026] | 0.042 |
| Qwen3-1.7B | 3 | 0.154 | surprise:+0.036[-0.001,+0.074]; fear:+0.030[-0.007,+0.066]; sadness:+0.020[-0.019,+0.063]; disgust:+0.015[-0.021,+0.049]; anger:+0.003[-0.033,+0.045]; happiness:-0.010[-0.050,+0.023] | 0.038 |
| Qwen3-4B | -1 | 0.000 | disgust:+0.001250;anger:+0.000000;fear:+0.000000;happiness:+0.000000;sadness:+0.000000;surprise:+0.000000 | 0.001 |
| Qwen3-4B | 1 | 0.676 | fear:+0.004[-0.037,+0.050]; surprise:-0.004[-0.050,+0.040]; happiness:-0.019[-0.060,+0.022]; sadness:-0.025[-0.069,+0.016]; anger:-0.037[-0.085,+0.004]; disgust:-0.066[-0.111,-0.021] | 0.060 |
| Qwen3-4B | 2 | 0.245 | sadness:+0.024[-0.021,+0.064]; disgust:+0.019[-0.024,+0.058]; happiness:+0.006[-0.035,+0.048]; anger:-0.001[-0.044,+0.036]; fear:-0.011[-0.050,+0.034]; surprise:-0.022[-0.062,+0.018] | 0.037 |
| Qwen3-4B | 3 | 0.079 | disgust:+0.048![+0.018,+0.076]; anger:+0.039[+0.011,+0.068]; surprise:+0.026[-0.001,+0.052]; happiness:+0.012[-0.015,+0.036]; fear:+0.007[-0.019,+0.034]; sadness:+0.001[-0.026,+0.030] | 0.039 |
| gemma-3-1b-it | -1 | 0.066 | anger:+0.011250;surprise:+0.007917;happiness:+0.006667;fear:+0.004167;sadness:+0.003750;disgust:+0.001667 | 0.010 |
| gemma-3-1b-it | 1 | 0.620 | sadness:+0.035000;happiness:+0.013333;surprise:+0.007083;anger:+0.005000;disgust:-0.003333;fear:-0.014583 | 0.050 |
| gemma-3-1b-it | 2 | 0.198 | fear:+0.005000;disgust:-0.005417;anger:-0.006250;happiness:-0.010833;surprise:-0.015000;sadness:-0.033750 | 0.039 |
| gemma-3-1b-it | 3 | 0.116 | disgust:+0.007083;fear:+0.005417;surprise:+0.000000;sadness:-0.005000;happiness:-0.009167;anger:-0.010000 | 0.017 |
| gemma-3-270m-it | -1 | 0.116 | fear:-0.001667;sadness:-0.007500;disgust:-0.009167;surprise:-0.011667;anger:-0.012500;happiness:-0.017917 | 0.016 |
| gemma-3-270m-it | 1 | 0.147 | fear:+0.025000;sadness:+0.016250;anger:+0.006250;disgust:+0.000833;surprise:-0.001667;happiness:-0.006250 | 0.031 |
| gemma-3-270m-it | 2 | 0.682 | anger:+0.011250;surprise:+0.009167;happiness:+0.001250;disgust:-0.006250;sadness:-0.015417;fear:-0.037083 | 0.048 |
| gemma-3-270m-it | 3 | 0.054 | happiness:+0.022917;disgust:+0.014583;fear:+0.013750;sadness:+0.006667;surprise:+0.004167;anger:-0.005000 | 0.028 |
| gemma-3-4b-it | -1 | 0.001 | anger:+0.002917;happiness:+0.002083;fear:+0.001667;sadness:+0.001250;disgust:+0.000417;surprise:+0.000000 | 0.003 |
| gemma-3-4b-it | 1 | 0.601 | surprise:+0.002083;disgust:-0.001250;anger:-0.007083;fear:-0.007083;happiness:-0.007083;sadness:-0.020000 | 0.022 |
| gemma-3-4b-it | 2 | 0.240 | happiness:+0.022917;surprise:+0.005417;sadness:+0.001250;fear:-0.002500;anger:-0.004167;disgust:-0.010833 | 0.034 |
| gemma-3-4b-it | 3 | 0.158 | sadness:+0.017500;disgust:+0.011667;anger:+0.008333;fear:+0.007917;surprise:-0.007500;happiness:-0.017917 | 0.035 |

### Ultimatum_Game_Proposer
| model | option_id | neutral | all emotion deltas (Δ vs neutral) | range |
|---|---|---:|---|---:|
| Llama-3.2-1B-Instruct | -1 | 0.010 | sadness:+0.443333;surprise:+0.282083;fear:+0.194167;disgust:+0.157917;anger:+0.102917;happiness:+0.073750 | 0.370 |
| Llama-3.2-1B-Instruct | 1 | 0.656 | anger:-0.069167;happiness:-0.217500;fear:-0.253750;disgust:-0.272917;sadness:-0.291250;surprise:-0.308750 | 0.240 |
| Llama-3.2-1B-Instruct | 2 | 0.179 | happiness:+0.090000;fear:+0.038750;disgust:+0.035417;surprise:+0.013750;anger:-0.033750;sadness:-0.100417 | 0.190 |
| Llama-3.2-1B-Instruct | 3 | 0.155 | disgust:+0.079583;happiness:+0.053750;fear:+0.020833;surprise:+0.012917;anger:-0.002083;sadness:-0.052083 | 0.132 |
| Llama-3.2-1B-Instruct | 4 | 0.000 | anger:+0.003750;sadness:+0.001250;disgust:+0.000000;fear:+0.000000;happiness:+0.000000;surprise:+0.000000 | 0.004 |
| Llama-3.2-1B-Instruct | 5 | 0.000 | anger:+0.001250;disgust:+0.000000;fear:+0.000000;happiness:+0.000000;sadness:+0.000000;surprise:+0.000000 | 0.001 |
| Llama-3.2-1B-Instruct | 6 | 0.000 | anger:+0.001250;disgust:+0.000000;fear:+0.000000;happiness:+0.000000;sadness:+0.000000;surprise:+0.000000 | 0.001 |
| Llama-3.2-3B-Instruct | -1 | 0.001 | surprise:+0.030000;anger:+0.026250;sadness:+0.020000;disgust:+0.010625;fear:+0.000417;happiness:-0.001250 | 0.031 |
| Llama-3.2-3B-Instruct | 1 | 0.491 |  | 0.274 |
| Llama-3.2-3B-Instruct | 2 | 0.296 |  | 0.172 |
| Llama-3.2-3B-Instruct | 3 | 0.211 |  | 0.109 |
| Phi-3.5-mini-instruct | -1 | 0.010 | disgust:-0.003333;sadness:-0.003750;fear:-0.004167;happiness:-0.005833;anger:-0.006250;surprise:-0.006667 | 0.003 |
| Phi-3.5-mini-instruct | 1 | 0.315 | sadness:+0.125!![+0.068,+0.189]; happiness:+0.092![+0.034,+0.161]; disgust:+0.048[-0.022,+0.124]; surprise:+0.038[-0.038,+0.102]; anger:+0.025[-0.034,+0.084]; fear:+0.005[-0.059,+0.064] | 0.043 |
| Phi-3.5-mini-instruct | 2 | 0.347 | disgust:-0.032[-0.145,+0.065]; fear:-0.054[-0.158,+0.045]; anger:-0.074[-0.153,+0.005]; sadness:-0.087[-0.170,-0.008]; surprise:-0.113[-0.215,+0.000]; happiness:-0.126![-0.207,-0.046] | 0.021 |
| Phi-3.5-mini-instruct | 3 | 0.328 | surprise:+0.075[-0.022,+0.172]; fear:+0.050[-0.050,+0.144]; anger:+0.049[-0.034,+0.138]; happiness:+0.034[-0.038,+0.111]; disgust:-0.016[-0.118,+0.086]; sadness:-0.038[-0.114,+0.038] | 0.065 |
| Phi-4-mini-instruct | -1 | 0.000 | sadness:+0.535417;disgust:+0.085417;anger:+0.025000;fear:+0.000000;happiness:+0.000000;surprise:+0.000000 | 0.535 |
| Phi-4-mini-instruct | 1 | 0.307 |  | 0.161 |
| Phi-4-mini-instruct | 2 | 0.321 |  | 0.196 |
| Phi-4-mini-instruct | 3 | 0.371 |  | 0.216 |
| Qwen2.5-0.5B-Instruct | -1 | 0.006 | sadness:+0.107500;surprise:+0.039167;anger:+0.026250;fear:+0.020417;disgust:+0.017917;happiness:-0.000417 | 0.108 |
| Qwen2.5-0.5B-Instruct | 1 | 0.540 | happiness:+0.126!![+0.057,+0.189]; disgust:+0.036[-0.027,+0.108]; sadness:-0.052[-0.126,+0.023]; surprise:-0.094![-0.156,-0.030]; fear:-0.199!!![-0.259,-0.136]; anger:-0.244!!![-0.302,-0.185] | 0.438 |
| Qwen2.5-0.5B-Instruct | 2 | 0.209 | anger:+0.185!!![+0.120,+0.253]; fear:+0.160!!![+0.094,+0.223]; disgust:-0.018[-0.075,+0.039]; surprise:-0.032[-0.089,+0.022]; sadness:-0.036[-0.097,+0.029]; happiness:-0.063[-0.111,-0.006] | 0.178 |
| Qwen2.5-0.5B-Instruct | 3 | 0.245 | surprise:+0.127!!![+0.067,+0.181]; sadness:+0.087![+0.019,+0.152]; anger:+0.058[-0.003,+0.117]; fear:+0.039[-0.016,+0.086]; disgust:-0.018[-0.078,+0.036]; happiness:-0.063[-0.123,-0.006] | 0.290 |
| Qwen2.5-1.5B-Instruct | -1 | 0.000 | disgust:+0.006667;sadness:+0.005417;anger:+0.004167;fear:+0.003750;happiness:+0.001250;surprise:+0.001250 | 0.005 |
| Qwen2.5-1.5B-Instruct | 1 | 0.564 | happiness:-0.037[-0.175,+0.113]; surprise:-0.040[-0.187,+0.093]; sadness:-0.070[-0.200,+0.043]; anger:-0.070[-0.210,+0.080]; fear:-0.080[-0.227,+0.067]; disgust:-0.241[-0.483,-0.034] | 0.096 |
| Qwen2.5-1.5B-Instruct | 2 | 0.255 | disgust:+0.138[-0.069,+0.345]; fear:+0.067[-0.093,+0.227]; anger:+0.000[-0.130,+0.110]; happiness:+0.000[-0.150,+0.125]; surprise:-0.027[-0.160,+0.080]; sadness:-0.052[-0.157,+0.052] | 0.089 |
| Qwen2.5-1.5B-Instruct | 3 | 0.181 | sadness:+0.122[+0.026,+0.217]; disgust:+0.103[-0.069,+0.276]; anger:+0.070[-0.040,+0.170]; surprise:+0.067[-0.040,+0.187]; happiness:+0.037[-0.087,+0.163]; fear:+0.013[-0.120,+0.133] | 0.053 |
| Qwen2.5-3B-Instruct | 1 | 0.466 | happiness:+0.174!!![+0.122,+0.223]; disgust:+0.130!!![+0.076,+0.186]; sadness:+0.000[-0.056,+0.058]; surprise:-0.049[-0.106,+0.000]; fear:-0.179!!![-0.237,-0.122]; anger:-0.218!!![-0.277,-0.162] | 0.300 |
| Qwen2.5-3B-Instruct | 2 | 0.275 | anger:+0.107!!![+0.059,+0.160]; fear:+0.058[+0.000,+0.104]; sadness:+0.000[-0.048,+0.044]; surprise:-0.009[-0.054,+0.040]; disgust:-0.017[-0.060,+0.031]; happiness:-0.050[-0.091,-0.004] | 0.096 |
| Qwen2.5-3B-Instruct | 3 | 0.259 | fear:+0.122!!![+0.068,+0.177]; anger:+0.111!!![+0.055,+0.172]; surprise:+0.058[+0.004,+0.110]; sadness:+0.000[-0.052,+0.052]; disgust:-0.113!!![-0.159,-0.066]; happiness:-0.124!!![-0.169,-0.079] | 0.215 |
| Qwen3-0.6B | -1 | 0.000 | disgust:+0.002083;anger:+0.000000;fear:+0.000000;happiness:+0.000000;sadness:+0.000000;surprise:+0.000000 | 0.002 |
| Qwen3-0.6B | 1 | 0.415 | sadness:+0.087[-0.022,+0.196]; anger:+0.071[-0.071,+0.214]; happiness:+0.030[-0.075,+0.119]; surprise:+0.000[-0.131,+0.119]; fear:-0.020[-0.196,+0.176]; disgust:-0.044[-0.132,+0.039] | 0.075 |
| Qwen3-0.6B | 2 | 0.350 | fear:-0.039[-0.255,+0.157]; disgust:-0.059[-0.146,+0.024]; happiness:-0.065[-0.159,+0.035]; surprise:-0.071[-0.214,+0.071]; sadness:-0.101[-0.217,+0.014]; anger:-0.179[-0.357,-0.036] | 0.055 |
| Qwen3-0.6B | 3 | 0.235 | anger:+0.107[+0.000,+0.232]; disgust:+0.102[+0.024,+0.176]; surprise:+0.071[-0.024,+0.167]; fear:+0.059[-0.118,+0.216]; happiness:+0.035[-0.050,+0.114]; sadness:+0.014[-0.058,+0.080] | 0.063 |
| Qwen3-1.7B | 1 | 0.515 | anger:+0.000[+0.000,+0.000]; disgust:+0.000[+0.000,+0.000]; fear:+0.000[+0.000,+0.000]; happiness:+0.000[+0.000,+0.000]; sadness:+0.000[+0.000,+0.000]; surprise:+0.000[+0.000,+0.000] | 0.039 |
| Qwen3-1.7B | 2 | 0.309 | anger:+0.000[+0.000,+0.000]; disgust:+0.000[+0.000,+0.000]; fear:+0.000[+0.000,+0.000]; happiness:+0.000[+0.000,+0.000]; sadness:+0.000[+0.000,+0.000]; surprise:+0.000[+0.000,+0.000] | 0.037 |
| Qwen3-1.7B | 3 | 0.176 | anger:+0.000[+0.000,+0.000]; disgust:+0.000[+0.000,+0.000]; fear:+0.000[+0.000,+0.000]; happiness:+0.000[+0.000,+0.000]; sadness:+0.000[+0.000,+0.000]; surprise:+0.000[+0.000,+0.000] | 0.063 |
| Qwen3-4B | 1 | 0.588 | anger:+0.093!![+0.041,+0.137]; happiness:+0.021[-0.027,+0.073]; fear:-0.007[-0.052,+0.038]; surprise:-0.012[-0.057,+0.030]; sadness:-0.041[-0.088,+0.004]; disgust:-0.045[-0.090,+0.003] | 0.096 |
| Qwen3-4B | 2 | 0.254 | surprise:+0.032[-0.008,+0.078]; disgust:+0.029[-0.018,+0.072]; fear:+0.013[-0.033,+0.058]; sadness:+0.005[-0.040,+0.052]; happiness:-0.008[-0.056,+0.037]; anger:-0.031[-0.078,+0.016] | 0.046 |
| Qwen3-4B | 3 | 0.159 | sadness:+0.036[-0.005,+0.072]; disgust:+0.016[-0.022,+0.051]; fear:-0.007[-0.041,+0.027]; happiness:-0.013[-0.054,+0.023]; surprise:-0.020[-0.058,+0.015]; anger:-0.062!![-0.093,-0.031] | 0.079 |
| gemma-3-1b-it | -1 | 0.003 | anger:+0.002500;fear:+0.002083;disgust:+0.001667;surprise:+0.001250;sadness:+0.000417;happiness:-0.001250 | 0.004 |
| gemma-3-1b-it | 1 | 0.711 | surprise:+0.017083;sadness:-0.010000;anger:-0.015417;disgust:-0.016250;fear:-0.019583;happiness:-0.040000 | 0.057 |
| gemma-3-1b-it | 2 | 0.181 | happiness:+0.015417;fear:+0.010833;disgust:-0.000833;anger:-0.005417;surprise:-0.009583;sadness:-0.013750 | 0.029 |
| gemma-3-1b-it | 3 | 0.105 | happiness:+0.025833;sadness:+0.023333;anger:+0.018333;disgust:+0.015417;fear:+0.006667;surprise:-0.006250 | 0.032 |
| gemma-3-270m-it | -1 | 0.116 | happiness:+0.020833;disgust:+0.007500;anger:+0.006667;sadness:+0.002917;fear:+0.002500;surprise:-0.002083 | 0.023 |
| gemma-3-270m-it | 1 | 0.235 | sadness:+0.028750;anger:+0.025833;surprise:+0.015833;disgust:+0.011667;fear:+0.011250;happiness:+0.005000 | 0.024 |
| gemma-3-270m-it | 2 | 0.560 | fear:-0.000417;surprise:-0.008750;disgust:-0.010000;sadness:-0.010000;happiness:-0.016667;anger:-0.017083 | 0.017 |
| gemma-3-270m-it | 3 | 0.089 | surprise:-0.005000;disgust:-0.009167;happiness:-0.009167;fear:-0.013333;anger:-0.015417;sadness:-0.021667 | 0.017 |
| gemma-3-4b-it | 1 | 0.853 | surprise:+0.004167;happiness:+0.003333;sadness:-0.011667;disgust:-0.013750;anger:-0.021667;fear:-0.022917 | 0.027 |
| gemma-3-4b-it | 2 | 0.104 | fear:+0.004167;anger:+0.001250;sadness:-0.005000;disgust:-0.007500;surprise:-0.012083;happiness:-0.012917 | 0.017 |
| gemma-3-4b-it | 3 | 0.044 | disgust:+0.021250;anger:+0.020417;fear:+0.018750;sadness:+0.016667;happiness:+0.009583;surprise:+0.007917 | 0.013 |

### Ultimatum_Game_Responder
| model | option_id | neutral | all emotion deltas (Δ vs neutral) | range |
|---|---|---:|---|---:|
| Llama-3.2-1B-Instruct | -1 | 0.007 | sadness:+0.402083;surprise:+0.262917;happiness:+0.065625;anger:+0.042500;disgust:-0.000833;fear:-0.001667 | 0.404 |
| Llama-3.2-1B-Instruct | 1 | 0.646 | anger:-0.133750;disgust:-0.140833;happiness:-0.169167;fear:-0.187917;surprise:-0.208333;sadness:-0.227500 | 0.094 |
| Llama-3.2-1B-Instruct | 19 | 0.000 | anger:+0.001250;disgust:+0.000000;fear:+0.000000;happiness:+0.000000;sadness:+0.000000;surprise:+0.000000 | 0.001 |
| Llama-3.2-1B-Instruct | 2 | 0.346 | fear:+0.189583;disgust:+0.141667;happiness:+0.127917;anger:+0.116250;surprise:-0.054583;sadness:-0.175417 | 0.365 |
| Llama-3.2-1B-Instruct | 3 | 0.000 | anger:+0.021250;sadness:+0.001250;disgust:+0.000000;fear:+0.000000;happiness:+0.000000;surprise:+0.000000 | 0.021 |
| Llama-3.2-1B-Instruct | 4 | 0.000 | anger:+0.001250;disgust:+0.000000;fear:+0.000000;happiness:+0.000000;sadness:+0.000000;surprise:+0.000000 | 0.001 |
| Llama-3.2-1B-Instruct | 6 | 0.000 | anger:+0.001250;disgust:+0.000000;fear:+0.000000;happiness:+0.000000;sadness:+0.000000;surprise:+0.000000 | 0.001 |
| Llama-3.2-3B-Instruct | -1 | 0.000 | sadness:+0.007500;anger:+0.005000;surprise:+0.003750;fear:+0.002500;disgust:+0.001875;happiness:+0.000000 | 0.007 |
| Llama-3.2-3B-Instruct | 1 | 0.486 | sadness:+0.160[-0.040,+0.360]; anger:+0.115[-0.154,+0.385]; disgust:+0.000[-0.429,+0.429]; surprise:-0.042[-0.208,+0.125]; fear:-0.211[-0.474,+0.053] | 0.184 |
| Llama-3.2-3B-Instruct | 2 | 0.514 | fear:+0.211[-0.053,+0.474]; surprise:+0.042[-0.125,+0.208]; disgust:+0.000[-0.429,+0.429]; anger:-0.115[-0.385,+0.154]; sadness:-0.160[-0.360,+0.040] | 0.185 |
| Phi-3.5-mini-instruct | -1 | 0.006 | disgust:+0.018333;anger:+0.008750;fear:+0.006250;surprise:-0.002917;sadness:-0.003333;happiness:-0.006250 | 0.025 |
| Phi-3.5-mini-instruct | 1 | 0.505 | anger:+0.045[-0.008,+0.102]; disgust:+0.040[-0.026,+0.095]; fear:+0.015[-0.037,+0.074]; happiness:+0.007[-0.053,+0.061]; sadness:-0.003[-0.058,+0.048]; surprise:-0.006[-0.064,+0.036] | 0.070 |
| Phi-3.5-mini-instruct | 2 | 0.489 | surprise:+0.006[-0.042,+0.064]; sadness:+0.003[-0.050,+0.056]; happiness:-0.007[-0.061,+0.053]; fear:-0.015[-0.074,+0.035]; disgust:-0.040[-0.095,+0.022]; anger:-0.045[-0.102,+0.005] | 0.081 |
| Phi-4-mini-instruct | -1 | 0.099 | sadness:+0.547500;anger:+0.309167;disgust:+0.287083;fear:-0.052917;surprise:-0.058333;happiness:-0.092500 | 0.640 |
| Phi-4-mini-instruct | 1 | 0.421 | sadness:+0.176[-0.059,+0.412]; happiness:+0.055[-0.045,+0.151]; fear:+0.040[-0.045,+0.126]; anger:+0.000[+0.000,+0.000]; disgust:+0.000[-0.167,+0.146]; surprise:+0.000[+0.000,+0.000] | 0.272 |
| Phi-4-mini-instruct | 2 | 0.480 | anger:+0.000[+0.000,+0.000]; disgust:+0.000[-0.146,+0.167]; surprise:+0.000[+0.000,+0.000]; fear:-0.040[-0.126,+0.045]; happiness:-0.055[-0.151,+0.045]; sadness:-0.176[-0.412,+0.059] | 0.384 |
| Phi-4-mini-instruct | 3 | 0.000 | sadness:+0.001250;anger:+0.000000;disgust:+0.000000;fear:+0.000000;happiness:+0.000000;surprise:+0.000000 | 0.001 |
| Qwen2.5-0.5B-Instruct | -1 | 0.000 | sadness:+0.024167;surprise:+0.003125;fear:+0.001250;happiness:+0.001250;anger:+0.000000;disgust:+0.000000 | 0.024 |
| Qwen2.5-0.5B-Instruct | 1 | 0.636 | disgust:-0.010[-0.058,+0.045]; fear:-0.068![-0.122,-0.014]; surprise:-0.095!!![-0.142,-0.045]; happiness:-0.098!!![-0.150,-0.045]; sadness:-0.137!!![-0.189,-0.077]; anger:-0.188!!![-0.241,-0.134] | 0.202 |
| Qwen2.5-0.5B-Instruct | 2 | 0.364 | anger:+0.188!!![+0.134,+0.241]; sadness:+0.137!!![+0.077,+0.187]; happiness:+0.098!!![+0.045,+0.148]; surprise:+0.095!!![+0.044,+0.141]; fear:+0.068![+0.012,+0.122]; disgust:+0.010[-0.045,+0.058] | 0.202 |
| Qwen2.5-1.5B-Instruct | -1 | 0.001 | sadness:+0.002500;fear:+0.000833;surprise:+0.000625;disgust:+0.000417;anger:+0.000000;happiness:-0.001250 | 0.004 |
| Qwen2.5-1.5B-Instruct | 1 | 0.874 | fear:-0.018[-0.140,+0.105]; disgust:-0.040[-0.120,+0.000]; anger:-0.083[-0.222,+0.056]; sadness:-0.101[-0.215,-0.013]; surprise:-0.104[-0.194,-0.030]; happiness:-0.114[-0.216,+0.000] | 0.093 |
| Qwen2.5-1.5B-Instruct | 2 | 0.125 | happiness:+0.114[+0.000,+0.216]; surprise:+0.104[+0.015,+0.194]; sadness:+0.101[+0.013,+0.203]; anger:+0.083[-0.056,+0.222]; disgust:+0.040[+0.000,+0.120]; fear:+0.018[-0.105,+0.140] | 0.092 |
| Qwen2.5-3B-Instruct | -1 | 0.001 | sadness:+0.003333;happiness:+0.002500;disgust:+0.000000;fear:+0.000000;anger:-0.001250;surprise:-0.001250 | 0.005 |
| Qwen2.5-3B-Instruct | 1 | 0.618 | anger:+0.143!!![+0.096,+0.188]; disgust:+0.072!![+0.028,+0.119]; fear:+0.058![+0.012,+0.104]; surprise:+0.018[-0.029,+0.064]; happiness:-0.045[-0.093,+0.001]; sadness:-0.050[-0.096,-0.003] | 0.154 |
| Qwen2.5-3B-Instruct | 2 | 0.381 | sadness:+0.050[+0.003,+0.096]; happiness:+0.045[-0.001,+0.090]; surprise:-0.018[-0.064,+0.029]; fear:-0.058![-0.105,-0.013]; disgust:-0.072!![-0.119,-0.029]; anger:-0.143!!![-0.188,-0.098] | 0.149 |
| Qwen3-0.6B | -1 | 0.000 | disgust:+0.001250;anger:+0.000000;fear:+0.000000;happiness:+0.000000;sadness:+0.000000;surprise:+0.000000 | 0.001 |
| Qwen3-0.6B | 1 | 0.463 | disgust:+0.088[+0.009,+0.154]; sadness:+0.082[-0.022,+0.201]; happiness:+0.058[-0.013,+0.132]; fear:+0.023[-0.070,+0.133]; surprise:+0.008[-0.102,+0.110]; anger:-0.106[-0.298,+0.085] | 0.080 |
| Qwen3-0.6B | 2 | 0.537 | anger:+0.106[-0.085,+0.298]; surprise:-0.008[-0.110,+0.102]; fear:-0.023[-0.133,+0.070]; happiness:-0.058[-0.132,+0.010]; sadness:-0.082[-0.201,+0.022]; disgust:-0.088[-0.154,-0.013] | 0.079 |
| Qwen3-1.7B | 1 | 0.542 | sadness:-0.008[-0.071,+0.050]; disgust:-0.010[-0.073,+0.042]; anger:-0.013[-0.075,+0.047]; fear:-0.014[-0.068,+0.041]; surprise:-0.017[-0.085,+0.036]; happiness:-0.020[-0.076,+0.039] | 0.036 |
| Qwen3-1.7B | 2 | 0.458 | happiness:+0.020[-0.039,+0.074]; surprise:+0.017[-0.036,+0.085]; fear:+0.014[-0.043,+0.068]; anger:+0.013[-0.047,+0.072]; disgust:+0.010[-0.042,+0.073]; sadness:+0.008[-0.052,+0.071] | 0.036 |
| Qwen3-4B | 1 | 0.517 | fear:+0.013[-0.035,+0.060]; surprise:+0.009[-0.045,+0.058]; anger:+0.008[-0.048,+0.055]; disgust:-0.003[-0.051,+0.040]; sadness:-0.016[-0.065,+0.028]; happiness:-0.021[-0.074,+0.029] | 0.031 |
| Qwen3-4B | 2 | 0.482 | happiness:+0.021[-0.034,+0.073]; sadness:+0.016[-0.029,+0.065]; disgust:+0.003[-0.041,+0.051]; anger:-0.008[-0.055,+0.044]; surprise:-0.009[-0.058,+0.044]; fear:-0.013[-0.061,+0.034] | 0.031 |
| gemma-3-1b-it | -1 | 0.000 | surprise:+0.001250;anger:+0.000000;disgust:+0.000000;fear:+0.000000;happiness:+0.000000;sadness:+0.000000 | 0.001 |
| gemma-3-1b-it | 1 | 0.774 | happiness:+0.017083;sadness:+0.008750;disgust:+0.008333;fear:+0.007917;anger:+0.002500;surprise:-0.021667 | 0.039 |
| gemma-3-1b-it | 2 | 0.226 | surprise:+0.021250;anger:-0.002500;fear:-0.007917;disgust:-0.008333;sadness:-0.008750;happiness:-0.017083 | 0.038 |
| gemma-3-270m-it | 1 | 0.453 | surprise:+0.016667;fear:+0.008750;sadness:+0.003333;happiness:-0.002083;anger:-0.011250;disgust:-0.024583 | 0.041 |
| gemma-3-270m-it | 2 | 0.547 | disgust:+0.024583;anger:+0.011250;happiness:+0.002083;sadness:-0.003333;fear:-0.008750;surprise:-0.016667 | 0.041 |
| gemma-3-4b-it | 1 | 0.644 | surprise:+0.048333;disgust:+0.034583;sadness:+0.034583;anger:+0.024167;fear:+0.015417;happiness:-0.003750 | 0.052 |
| gemma-3-4b-it | 2 | 0.356 | happiness:+0.003750;fear:-0.015417;anger:-0.024167;disgust:-0.034583;sadness:-0.034583;surprise:-0.048333 | 0.052 |

## Strongest Behavior Effects (Top 20 by delta_range)
| game_setting | model | behavior_label | neutral | best (Δ) | worst (Δ) | range |
|---|---|---|---:|---|---|---:|
| Ultimatum_Game_Responder | Qwen2.5-0.5B-Instruct | reject | 0.362 | anger (+0.574) | happiness (-0.265) | 0.838 |
| Ultimatum_Game_Responder | Qwen2.5-0.5B-Instruct | accept | 0.637 | happiness (+0.264) | anger (-0.574) | 0.838 |
| Ultimatum_Game_Responder | Llama-3.2-1B-Instruct | reject | 0.336 | fear (+0.550) | happiness (-0.279) | 0.829 |
| Ultimatum_Game_Responder | Llama-3.2-1B-Instruct | accept | 0.656 | happiness (+0.237) | fear (-0.549) | 0.786 |
| Ultimatum_Game_Responder | Phi-4-mini-instruct | accept | 0.715 | happiness (+0.260) | sadness (-0.500) | 0.760 |
| Ultimatum_Game_Responder | Llama-3.2-3B-Instruct | accept | 0.931 | happiness (+0.067) | fear (-0.620) | 0.687 |
| Ultimatum_Game_Responder | Llama-3.2-3B-Instruct | reject | 0.069 | fear (+0.618) | happiness (-0.064) | 0.682 |
| Ultimatum_Game_Responder | Phi-4-mini-instruct | unknown | 0.099 | sadness (+0.548) | happiness (-0.092) | 0.640 |
| Trust_Game_Trustee | Llama-3.2-1B-Instruct | unknown | 0.095 | sadness (+0.713) | fear (+0.080) | 0.633 |
| Trust_Game_Trustee | Phi-4-mini-instruct | unknown | 0.014 | sadness (+0.618) | happiness (-0.001) | 0.619 |
| Trust_Game_Trustor | Llama-3.2-3B-Instruct | trust_high | 0.701 | happiness (+0.070) | surprise (-0.520) | 0.590 |
| Ultimatum_Game_Proposer | Phi-4-mini-instruct | unknown | 0.000 | sadness (+0.535) | fear (+0.000) | 0.535 |
| Trust_Game_Trustor | Llama-3.2-1B-Instruct | unknown | 0.152 | sadness (+0.732) | disgust (+0.229) | 0.503 |
| Trust_Game_Trustee | Llama-3.2-3B-Instruct | return_high | 0.681 | happiness (+0.030) | disgust (-0.469) | 0.499 |
| Trust_Game_Trustor | Llama-3.2-3B-Instruct | trust_none | 0.011 | disgust (+0.460) | happiness (-0.009) | 0.469 |
| Ultimatum_Game_Proposer | Phi-4-mini-instruct | offer_high | 0.845 | happiness (-0.006) | sadness (-0.472) | 0.466 |
| Trust_Game_Trustee | Phi-4-mini-instruct | return_high | 0.584 | happiness (+0.083) | sadness (-0.373) | 0.455 |
| Trust_Game_Trustee | Llama-3.2-3B-Instruct | return_none | 0.025 | disgust (+0.421) | happiness (-0.022) | 0.444 |
| Trust_Game_Trustor | Phi-4-mini-instruct | trust_high | 0.349 | happiness (+0.178) | disgust (-0.261) | 0.439 |
| Trust_Game_Trustor | Phi-4-mini-instruct | unknown | 0.001 | sadness (+0.416) | happiness (+0.001) | 0.414 |

## Per Game Setting Behavior (All models)
### Trust_Game_Trustee
| model | behavior_label | neutral | all emotion deltas (Δ vs neutral) | range |
|---|---|---:|---|---:|
| Llama-3.2-1B-Instruct | return_high | 0.334 | anger:-0.107083;disgust:-0.114583;fear:-0.132917;happiness:-0.147917;surprise:-0.231875;sadness:-0.236250 | 0.129 |
| Llama-3.2-1B-Instruct | return_medium | 0.511 | fear:-0.015833;disgust:-0.148750;happiness:-0.194167;anger:-0.210417;sadness:-0.342500;surprise:-0.364167 | 0.348 |
| Llama-3.2-1B-Instruct | return_none | 0.060 | disgust:+0.093333;fear:+0.068750;anger:+0.015000;happiness:-0.014167;surprise:-0.032083;sadness:-0.038750 | 0.132 |
| Llama-3.2-1B-Instruct | unknown | 0.095 | sadness:+0.713333;surprise:+0.662083;happiness:+0.356250;anger:+0.302500;disgust:+0.170000;fear:+0.080000 | 0.633 |
| Llama-3.2-3B-Instruct | return_high | 0.681 | sadness:+1.000[+1.000,+1.000]; fear:+0.000[-1.000,+1.000]; surprise:+0.000[+0.000,+0.000]; anger:-0.500[-1.000,+0.000] | 0.499 |
| Llama-3.2-3B-Instruct | return_medium | 0.207 | anger:+0.500[+0.000,+1.000]; surprise:+0.000[+0.000,+0.000]; fear:-0.500[-1.000,+0.000]; sadness:-1.000[-1.000,-1.000] | 0.268 |
| Llama-3.2-3B-Instruct | return_none | 0.025 | fear:+0.500[+0.000,+1.000]; anger:+0.000[+0.000,+0.000]; sadness:+0.000[+0.000,+0.000]; surprise:+0.000[+0.000,+0.000] | 0.444 |
| Llama-3.2-3B-Instruct | unknown | 0.086 | surprise:+0.255833;sadness:+0.249583;happiness:+0.027500;anger:+0.000417;disgust:+0.000000;fear:-0.024583 | 0.280 |
| Phi-3.5-mini-instruct | return_high | 0.537 | fear:+0.068!![+0.023,+0.110]; surprise:+0.031[-0.011,+0.073]; happiness:+0.012[-0.026,+0.053]; anger:+0.007[-0.032,+0.049]; sadness:-0.087!!![-0.119,-0.051]; disgust:-0.115!!![-0.165,-0.074] | 0.155 |
| Phi-3.5-mini-instruct | return_medium | 0.453 | disgust:+0.115!!![+0.074,+0.165]; sadness:+0.087!!![+0.051,+0.119]; anger:-0.007[-0.049,+0.032]; happiness:-0.012[-0.053,+0.025]; surprise:-0.031[-0.073,+0.009]; fear:-0.068!![-0.110,-0.023] | 0.151 |
| Phi-3.5-mini-instruct | return_none | 0.000 | anger:+0.001250;disgust:+0.001250;fear:+0.001250;happiness:+0.000000;sadness:+0.000000;surprise:+0.000000 | 0.001 |
| Phi-3.5-mini-instruct | unknown | 0.010 | disgust:+0.003750;sadness:+0.002500;anger:-0.001667;fear:-0.001667;surprise:-0.005000;happiness:-0.007083 | 0.011 |
| Phi-4-mini-instruct | return_high | 0.584 | disgust:+0.500[+0.000,+1.000]; happiness:+0.333[+0.000,+1.000] | 0.455 |
| Phi-4-mini-instruct | return_medium | 0.399 | happiness:-0.333[-1.000,+0.000]; disgust:-0.500[-1.000,+0.000] | 0.399 |
| Phi-4-mini-instruct | return_none | 0.004 | disgust:+0.005417;anger:+0.002917;sadness:+0.001250;fear:-0.000417;surprise:-0.001875;happiness:-0.003750 | 0.009 |
| Phi-4-mini-instruct | unknown | 0.014 | sadness:+0.617917;disgust:+0.121667;anger:+0.059583;fear:+0.018750;surprise:+0.008333;happiness:-0.000833 | 0.619 |
| Qwen2.5-0.5B-Instruct | return_high | 0.346 | fear:+0.020[-0.039,+0.081]; surprise:-0.014[-0.082,+0.052]; happiness:-0.018[-0.109,+0.050]; anger:-0.025[-0.113,+0.046]; sadness:-0.108![-0.187,-0.036]; disgust:-0.129!!![-0.186,-0.068] | 0.187 |
| Qwen2.5-0.5B-Instruct | return_medium | 0.438 | anger:+0.084[+0.000,+0.172]; fear:+0.083![+0.024,+0.142]; disgust:-0.018[-0.100,+0.052]; happiness:-0.032[-0.109,+0.044]; surprise:-0.044[-0.114,+0.027]; sadness:-0.092[-0.163,-0.016] | 0.125 |
| Qwen2.5-0.5B-Instruct | return_none | 0.184 | sadness:+0.199!!![+0.135,+0.275]; disgust:+0.147!!![+0.087,+0.192]; surprise:+0.057[+0.000,+0.109]; happiness:+0.050[-0.009,+0.106]; anger:-0.059[-0.113,-0.008]; fear:-0.103!!![-0.144,-0.072] | 0.194 |
| Qwen2.5-0.5B-Instruct | unknown | 0.033 | sadness:+0.129167;disgust:+0.081667;happiness:+0.055000;surprise:+0.042917;anger:+0.041667;fear:-0.003750 | 0.133 |
| Qwen2.5-1.5B-Instruct | return_high | 0.314 | happiness:+0.078[-0.098,+0.255]; surprise:-0.068[-0.227,+0.136]; anger:-0.161[-0.339,+0.018]; sadness:-0.171[-0.314,-0.043]; fear:-0.267[-0.422,-0.111]; disgust:-0.286[-0.714,+0.286] | 0.149 |
| Qwen2.5-1.5B-Instruct | return_medium | 0.588 | fear:+0.200[+0.022,+0.356]; anger:+0.161[+0.000,+0.339]; sadness:+0.157[+0.000,+0.286]; surprise:+0.091[-0.091,+0.250]; happiness:-0.059[-0.235,+0.118]; disgust:-0.143[-0.571,+0.286] | 0.089 |
| Qwen2.5-1.5B-Instruct | return_none | 0.087 | disgust:+0.429[+0.143,+0.857]; fear:+0.067[-0.044,+0.178]; sadness:+0.014[-0.086,+0.114]; anger:+0.000[-0.071,+0.071]; happiness:-0.020[-0.098,+0.078]; surprise:-0.023[-0.136,+0.068] | 0.122 |
| Qwen2.5-1.5B-Instruct | unknown | 0.011 | fear:+0.012083;disgust:+0.005417;anger:+0.002500;happiness:+0.002083;sadness:+0.002083;surprise:-0.000833 | 0.013 |
| Qwen2.5-3B-Instruct | return_high | 0.384 | happiness:+0.053[+0.000,+0.099]; surprise:+0.016[-0.024,+0.065]; sadness:+0.008[-0.036,+0.051]; anger:-0.020[-0.071,+0.019]; fear:-0.023[-0.067,+0.022]; disgust:-0.050[-0.090,-0.004] | 0.083 |
| Qwen2.5-3B-Instruct | return_medium | 0.534 | fear:+0.013[-0.037,+0.058]; anger:+0.009[-0.033,+0.060]; sadness:-0.013[-0.057,+0.034]; surprise:-0.030[-0.081,+0.019]; disgust:-0.031[-0.079,+0.015]; happiness:-0.062[-0.109,-0.013] | 0.050 |
| Qwen2.5-3B-Instruct | return_none | 0.079 | disgust:+0.080!!![+0.047,+0.108]; surprise:+0.014[-0.014,+0.038]; anger:+0.011[-0.011,+0.038]; fear:+0.010[-0.018,+0.037]; happiness:+0.009[-0.019,+0.034]; sadness:+0.005[-0.022,+0.030] | 0.067 |
| Qwen2.5-3B-Instruct | unknown | 0.004 | sadness:-0.000833;fear:-0.001667;happiness:-0.001667;disgust:-0.002083;anger:-0.002500;surprise:-0.002500 | 0.002 |
| Qwen3-0.6B | return_high | 0.579 | happiness:+0.059[+0.011,+0.104]; sadness:+0.042[-0.007,+0.096]; fear:-0.021[-0.092,+0.050]; disgust:-0.027[-0.087,+0.027]; surprise:-0.035[-0.093,+0.022]; anger:-0.049[-0.123,+0.031] | 0.122 |
| Qwen3-0.6B | return_medium | 0.412 | anger:+0.056[-0.037,+0.130]; surprise:+0.045[-0.010,+0.096]; fear:+0.021[-0.055,+0.088]; disgust:+0.016[-0.035,+0.076]; sadness:-0.032[-0.089,+0.017]; happiness:-0.053[-0.099,-0.007] | 0.111 |
| Qwen3-0.6B | return_none | 0.009 | disgust:+0.011[+0.003,+0.022]; fear:+0.000[-0.013,+0.013]; happiness:-0.005[-0.016,+0.004]; anger:-0.006[-0.025,+0.012]; surprise:-0.010[-0.022,+0.003]; sadness:-0.010[-0.020,-0.002] | 0.009 |
| Qwen3-0.6B | unknown | 0.000 | anger:+0.006250;fear:+0.006250;disgust:+0.005417;happiness:+0.002500;surprise:+0.001667;sadness:+0.000000 | 0.006 |
| Qwen3-1.7B | return_high | 0.600 | fear:-0.023[-0.086,+0.032]; disgust:-0.032[-0.089,+0.028]; happiness:-0.050[-0.104,+0.000]; sadness:-0.077!![-0.118,-0.037]; anger:-0.088!![-0.135,-0.036]; surprise:-0.109!![-0.162,-0.056] | 0.050 |
| Qwen3-1.7B | return_medium | 0.398 | surprise:+0.109!![+0.053,+0.158]; anger:+0.088!![+0.036,+0.135]; sadness:+0.077!![+0.037,+0.114]; happiness:+0.050[-0.004,+0.104]; disgust:+0.024[-0.036,+0.081]; fear:+0.018[-0.041,+0.077] | 0.056 |
| Qwen3-1.7B | return_none | 0.001 | disgust:+0.008[+0.000,+0.020]; fear:+0.005[+0.000,+0.018]; anger:+0.000[+0.000,+0.000]; happiness:+0.000[+0.000,+0.000]; sadness:+0.000[+0.000,+0.000]; surprise:+0.000[+0.000,+0.000] | 0.010 |
| Qwen3-1.7B | unknown | 0.001 | disgust:+0.000000;happiness:+0.000000;sadness:+0.000000;surprise:+0.000000;anger:-0.001250;fear:-0.001250 | 0.001 |
| Qwen3-4B | return_high | 0.743 | disgust:+0.000[-0.041,+0.039]; anger:-0.011[-0.054,+0.033]; fear:-0.016[-0.060,+0.024]; happiness:-0.030[-0.069,+0.014]; sadness:-0.040[-0.081,+0.001]; surprise:-0.041[-0.088,+0.000] | 0.049 |
| Qwen3-4B | return_medium | 0.255 | surprise:+0.039[-0.003,+0.081]; sadness:+0.036[-0.005,+0.076]; happiness:+0.029[-0.015,+0.069]; fear:+0.011[-0.029,+0.055]; anger:+0.009[-0.034,+0.049]; disgust:-0.013[-0.055,+0.024] | 0.054 |
| Qwen3-4B | return_none | 0.003 | disgust:+0.013[+0.005,+0.023]; fear:+0.005[-0.001,+0.013]; sadness:+0.004[-0.003,+0.010]; anger:+0.003[-0.004,+0.009]; surprise:+0.003[-0.003,+0.009]; happiness:+0.001[-0.004,+0.006] | 0.007 |
| gemma-3-1b-it | return_high | 0.489 | fear:+0.018750;anger:+0.006250;disgust:+0.005833;sadness:-0.006250;happiness:-0.007500;surprise:-0.008333 | 0.027 |
| gemma-3-1b-it | return_medium | 0.460 | surprise:+0.006250;sadness:+0.001250;happiness:+0.000417;anger:-0.010000;disgust:-0.011250;fear:-0.029167 | 0.035 |
| gemma-3-1b-it | return_none | 0.009 | fear:+0.002083;anger:-0.000417;disgust:-0.000833;happiness:-0.001667;sadness:-0.002917;surprise:-0.002917 | 0.005 |
| gemma-3-1b-it | unknown | 0.043 | happiness:+0.008750;fear:+0.008333;sadness:+0.007917;disgust:+0.006250;surprise:+0.005000;anger:+0.004167 | 0.005 |
| gemma-3-270m-it | return_high | 0.274 | fear:+0.006667;sadness:+0.003333;happiness:-0.005417;disgust:-0.007500;anger:-0.023750;surprise:-0.029583 | 0.036 |
| gemma-3-270m-it | return_medium | 0.246 | surprise:+0.012083;fear:+0.005417;anger:+0.002917;sadness:-0.002917;disgust:-0.023333;happiness:-0.030833 | 0.043 |
| gemma-3-270m-it | return_none | 0.081 | happiness:+0.024167;disgust:+0.017083;anger:+0.011250;surprise:+0.009583;sadness:-0.005417;fear:-0.012917 | 0.037 |
| gemma-3-270m-it | unknown | 0.399 | disgust:+0.013750;happiness:+0.012083;anger:+0.009583;surprise:+0.007917;sadness:+0.005000;fear:+0.000833 | 0.013 |
| gemma-3-4b-it | return_high | 0.550 | fear:+0.021667;happiness:+0.010833;surprise:-0.001250;disgust:-0.012500;sadness:-0.013333;anger:-0.015833 | 0.037 |
| gemma-3-4b-it | return_medium | 0.340 | anger:+0.021667;disgust:+0.020000;sadness:+0.020000;surprise:+0.001250;happiness:-0.003333;fear:-0.016667 | 0.038 |
| gemma-3-4b-it | return_none | 0.015 | surprise:+0.000000;happiness:-0.003333;anger:-0.003750;fear:-0.003750;sadness:-0.005000;disgust:-0.005417 | 0.005 |
| gemma-3-4b-it | unknown | 0.095 | surprise:+0.000000;fear:-0.001250;sadness:-0.001667;anger:-0.002083;disgust:-0.002083;happiness:-0.004167 | 0.004 |

#### Item Change vs Neutral (paired by item_id, ignores intensity)
| model | change rates (emotion:%, n) |
|---|---|
| Llama-3.2-1B-Instruct | happiness:55.3% (n=674); disgust:55.0% (n=694); fear:51.0% (n=686); surprise:49.1% (n=432); anger:48.2% (n=695); sadness:45.3% (n=382) |
| Llama-3.2-3B-Instruct | disgust:88.0% (n=728); anger:68.0% (n=726); surprise:45.8% (n=721); fear:42.7% (n=723); sadness:36.4% (n=726); happiness:34.9% (n=720) |
| Phi-3.5-mini-instruct | disgust:32.4% (n=790); fear:29.2% (n=792); anger:27.7% (n=792); sadness:26.6% (n=790); happiness:26.3% (n=792); surprise:26.0% (n=792) |
| Phi-4-mini-instruct | disgust:41.1% (n=781); anger:35.6% (n=781); surprise:33.8% (n=783); sadness:32.9% (n=583); fear:30.0% (n=783); happiness:28.9% (n=789) |
| Qwen2.5-0.5B-Instruct | surprise:63.7% (n=766); happiness:62.1% (n=765); disgust:61.7% (n=767); sadness:60.3% (n=769); anger:58.2% (n=770); fear:55.7% (n=770) |
| Qwen2.5-1.5B-Instruct | fear:50.8% (n=791); disgust:49.9% (n=790); sadness:49.8% (n=791); happiness:49.1% (n=790); anger:46.6% (n=791); surprise:45.2% (n=790) |
| Qwen2.5-3B-Instruct | surprise:54.1% (n=797); disgust:54.0% (n=797); happiness:52.5% (n=796); fear:48.1% (n=797); sadness:47.1% (n=796); anger:46.8% (n=797) |
| Qwen3-0.6B | anger:38.0% (n=799); surprise:33.8% (n=799); sadness:33.8% (n=800); disgust:32.9% (n=799); fear:32.6% (n=798); happiness:32.4% (n=800) |
| Qwen3-1.7B | disgust:32.3% (n=799); surprise:32.1% (n=798); fear:32.0% (n=799); happiness:31.2% (n=799); sadness:29.0% (n=799); anger:28.8% (n=799) |
| Qwen3-4B | sadness:37.4% (n=800); happiness:36.6% (n=800); surprise:35.4% (n=800); fear:33.4% (n=800); anger:33.2% (n=800); disgust:31.6% (n=800) |
| gemma-3-1b-it | sadness:37.8% (n=760); happiness:37.1% (n=757); surprise:35.8% (n=755); anger:35.6% (n=756); fear:35.5% (n=761); disgust:34.0% (n=761) |
| gemma-3-270m-it | anger:54.5% (n=404); happiness:52.4% (n=403); sadness:51.4% (n=416); disgust:50.7% (n=428); surprise:48.9% (n=407); fear:46.5% (n=413) |
| gemma-3-4b-it | surprise:42.0% (n=722); disgust:40.7% (n=723); sadness:39.9% (n=721); fear:39.4% (n=723); happiness:37.6% (n=721); anger:37.1% (n=722) |

### Trust_Game_Trustor
| model | behavior_label | neutral | all emotion deltas (Δ vs neutral) | range |
|---|---|---:|---|---:|
| Llama-3.2-1B-Instruct | trust_high | 0.282 | anger:-0.096667;disgust:-0.106667;happiness:-0.146250;fear:-0.173333;surprise:-0.188125;sadness:-0.228750 | 0.132 |
| Llama-3.2-1B-Instruct | trust_low | 0.484 | fear:-0.115000;disgust:-0.175417;happiness:-0.187500;anger:-0.205833;surprise:-0.309583;sadness:-0.373750 | 0.259 |
| Llama-3.2-1B-Instruct | trust_none | 0.081 | disgust:+0.053333;fear:+0.051667;anger:+0.012917;happiness:-0.040000;surprise:-0.050000;sadness:-0.071875 | 0.125 |
| Llama-3.2-1B-Instruct | unknown | 0.152 | sadness:+0.732083;surprise:+0.589583;happiness:+0.373750;anger:+0.289583;fear:+0.236667;disgust:+0.228750 | 0.503 |
| Llama-3.2-3B-Instruct | trust_high | 0.701 | fear:-0.333[-1.000,+0.000] | 0.590 |
| Llama-3.2-3B-Instruct | trust_low | 0.214 | fear:+0.333[+0.000,+1.000] | 0.356 |
| Llama-3.2-3B-Instruct | trust_none | 0.011 | fear:+0.000[+0.000,+0.000] | 0.469 |
| Llama-3.2-3B-Instruct | unknown | 0.074 | surprise:+0.330417;sadness:+0.303333;fear:+0.117500;anger:+0.033333;happiness:+0.006667;disgust:-0.002500 | 0.333 |
| Phi-3.5-mini-instruct | trust_high | 0.389 | fear:+0.049[+0.013,+0.084]; anger:+0.012[-0.028,+0.050]; surprise:+0.011[-0.031,+0.051]; happiness:-0.012[-0.053,+0.033]; sadness:-0.046[-0.086,-0.008]; disgust:-0.048[-0.091,-0.007] | 0.074 |
| Phi-3.5-mini-instruct | trust_low | 0.609 | sadness:+0.042[+0.003,+0.081]; disgust:+0.041[-0.003,+0.083]; happiness:+0.013[-0.035,+0.053]; surprise:-0.011[-0.049,+0.031]; anger:-0.015[-0.056,+0.024]; fear:-0.053[-0.091,-0.017] | 0.071 |
| Phi-3.5-mini-instruct | trust_none | 0.001 | disgust:+0.007[+0.001,+0.013]; fear:+0.004[+0.000,+0.009]; sadness:+0.004[+0.000,+0.009]; anger:+0.003[+0.000,+0.007]; surprise:+0.000[-0.004,+0.004]; happiness:-0.001[-0.004,+0.000] | 0.006 |
| Phi-3.5-mini-instruct | unknown | 0.001 | surprise:+0.003333;sadness:+0.002917;disgust:+0.002083;anger:+0.001667;fear:+0.001250;happiness:+0.000417 | 0.003 |
| Phi-4-mini-instruct | trust_high | 0.349 | happiness:+0.248!!![+0.199,+0.296]; fear:+0.144!!![+0.104,+0.186]; surprise:-0.046[-0.161,+0.080]; sadness:-0.085!!![-0.124,-0.043]; anger:-0.139!![-0.232,-0.053]; disgust:-0.229!!![-0.270,-0.186] | 0.439 |
| Phi-4-mini-instruct | trust_low | 0.645 | disgust:+0.214!!![+0.163,+0.259]; anger:+0.139!![+0.053,+0.232]; sadness:+0.059![+0.016,+0.102]; surprise:+0.046[-0.080,+0.149]; fear:-0.148!!![-0.189,-0.111]; happiness:-0.246!!![-0.296,-0.197] | 0.378 |
| Phi-4-mini-instruct | trust_none | 0.005 | sadness:+0.026!!![+0.010,+0.040]; disgust:+0.015[+0.004,+0.030]; fear:+0.003[-0.003,+0.010]; anger:+0.000[+0.000,+0.000]; surprise:+0.000[+0.000,+0.000]; happiness:-0.002[-0.006,+0.000] | 0.042 |
| Phi-4-mini-instruct | unknown | 0.001 | sadness:+0.415625;disgust:+0.020417;anger:+0.019167;surprise:+0.006250;fear:+0.002500;happiness:+0.001250 | 0.414 |
| Qwen2.5-0.5B-Instruct | trust_high | 0.304 | fear:+0.092[+0.010,+0.174]; happiness:-0.003[-0.067,+0.061]; disgust:-0.037[-0.121,+0.042]; anger:-0.038[-0.205,+0.103]; surprise:-0.042[-0.110,+0.020]; sadness:-0.143!!![-0.202,-0.089] | 0.234 |
| Qwen2.5-0.5B-Instruct | trust_low | 0.376 | anger:+0.038[-0.115,+0.179]; fear:+0.005[-0.077,+0.092]; surprise:+0.000[-0.074,+0.065]; happiness:-0.085[-0.155,-0.015]; disgust:-0.104![-0.183,-0.013]; sadness:-0.266!!![-0.328,-0.209] | 0.200 |
| Qwen2.5-0.5B-Instruct | trust_none | 0.146 | sadness:+0.409!!![+0.355,+0.468]; disgust:+0.142!!![+0.062,+0.204]; happiness:+0.088![+0.024,+0.146]; surprise:+0.042[-0.014,+0.096]; anger:+0.000[-0.090,+0.103]; fear:-0.097![-0.154,-0.031] | 0.304 |
| Qwen2.5-0.5B-Instruct | unknown | 0.174 | happiness:+0.163750;surprise:+0.015833;disgust:+0.001250;sadness:-0.019583;anger:-0.074167;fear:-0.118750 | 0.282 |
| Qwen2.5-1.5B-Instruct | trust_high | 0.386 | surprise:+0.080!![+0.028,+0.124]; happiness:+0.020[-0.031,+0.061]; disgust:-0.029[-0.079,+0.023]; sadness:-0.091!!![-0.143,-0.039]; fear:-0.110!!![-0.155,-0.062]; anger:-0.133!!![-0.174,-0.091] | 0.175 |
| Qwen2.5-1.5B-Instruct | trust_low | 0.521 | anger:+0.096!!![+0.049,+0.137]; fear:+0.082!![+0.033,+0.129]; sadness:+0.064![+0.011,+0.110]; happiness:-0.028[-0.073,+0.023]; disgust:-0.052[-0.104,+0.007]; surprise:-0.055![-0.100,-0.005] | 0.144 |
| Qwen2.5-1.5B-Instruct | trust_none | 0.079 | disgust:+0.081!!![+0.045,+0.113]; anger:+0.037![+0.008,+0.066]; fear:+0.028[-0.001,+0.056]; sadness:+0.026[+0.001,+0.053]; happiness:+0.008[-0.020,+0.035]; surprise:-0.025[-0.047,-0.003] | 0.043 |
| Qwen2.5-1.5B-Instruct | unknown | 0.014 | fear:+0.006250;sadness:+0.004583;anger:+0.002917;surprise:+0.002917;happiness:-0.000417;disgust:-0.003333 | 0.010 |
| Qwen2.5-3B-Instruct | trust_high | 0.294 | happiness:+0.078!![+0.034,+0.117]; surprise:+0.044[+0.000,+0.088]; disgust:+0.011[-0.031,+0.058]; sadness:-0.033[-0.077,+0.005]; fear:-0.043[-0.083,-0.008]; anger:-0.048[-0.092,-0.010] | 0.092 |
| Qwen2.5-3B-Instruct | trust_low | 0.609 | anger:+0.081!![+0.039,+0.132]; fear:+0.065![+0.026,+0.112]; sadness:+0.035[-0.013,+0.082]; surprise:+0.003[-0.043,+0.052]; happiness:-0.088!![-0.142,-0.043]; disgust:-0.128!!![-0.176,-0.082] | 0.133 |
| Qwen2.5-3B-Instruct | trust_none | 0.095 | disgust:+0.117!!![+0.086,+0.152]; happiness:+0.010[-0.023,+0.039]; sadness:-0.003[-0.029,+0.025]; fear:-0.023[-0.050,+0.003]; anger:-0.033![-0.058,-0.009]; surprise:-0.047!![-0.073,-0.024] | 0.113 |
| Qwen2.5-3B-Instruct | unknown | 0.003 | anger:+0.001875;happiness:+0.001250;fear:+0.000833;disgust:-0.000417;sadness:-0.000417;surprise:-0.000417 | 0.002 |
| Qwen3-0.6B | trust_high | 0.706 | happiness:+0.050[+0.002,+0.097]; disgust:+0.004[-0.049,+0.047]; fear:-0.071[-0.125,-0.008]; sadness:-0.090![-0.159,-0.031]; surprise:-0.160!![-0.242,-0.088]; anger:-0.340!![-0.520,-0.180] | 0.094 |
| Qwen3-0.6B | trust_low | 0.281 | anger:+0.260![+0.120,+0.440]; surprise:+0.103![+0.031,+0.180]; sadness:+0.079![+0.017,+0.148]; fear:+0.067[+0.008,+0.122]; disgust:-0.008[-0.053,+0.039]; happiness:-0.054[-0.101,-0.010] | 0.055 |
| Qwen3-0.6B | trust_none | 0.011 | anger:+0.080[+0.020,+0.160]; surprise:+0.057!![+0.026,+0.088]; sadness:+0.010[-0.007,+0.028]; disgust:+0.004[-0.006,+0.016]; fear:+0.004[-0.012,+0.020]; happiness:+0.004[-0.006,+0.017] | 0.060 |
| Qwen3-0.6B | unknown | 0.001 | fear:+0.005000;happiness:+0.002500;disgust:+0.000833;surprise:+0.000000;anger:-0.001250;sadness:-0.001250 | 0.006 |
| Qwen3-1.7B | trust_high | 0.555 | fear:+0.030[-0.018,+0.069]; disgust:+0.019[-0.029,+0.062]; happiness:+0.016[-0.034,+0.057]; anger:+0.004[-0.041,+0.049]; surprise:-0.014[-0.061,+0.030]; sadness:-0.026[-0.073,+0.017] | 0.052 |
| Qwen3-1.7B | trust_low | 0.431 | sadness:+0.020[-0.020,+0.066]; surprise:+0.013[-0.033,+0.061]; anger:-0.006[-0.054,+0.039]; happiness:-0.007[-0.050,+0.041]; disgust:-0.024[-0.068,+0.024]; fear:-0.027[-0.069,+0.019] | 0.053 |
| Qwen3-1.7B | trust_none | 0.014 | sadness:+0.006[-0.006,+0.015]; disgust:+0.004[-0.006,+0.015]; anger:+0.001[-0.010,+0.013]; surprise:+0.001[-0.007,+0.012]; fear:-0.003[-0.013,+0.004]; happiness:-0.009[-0.018,-0.001] | 0.012 |
| Qwen3-4B | trust_high | 0.557 | anger:+0.088!![+0.044,+0.133]; disgust:+0.049[+0.001,+0.092]; surprise:+0.015[-0.031,+0.059]; fear:+0.000[-0.045,+0.044]; happiness:-0.012[-0.059,+0.028]; sadness:-0.050[-0.098,-0.001] | 0.105 |
| Qwen3-4B | trust_low | 0.441 | sadness:+0.046[-0.001,+0.092]; happiness:+0.014[-0.028,+0.060]; fear:-0.003[-0.049,+0.041]; surprise:-0.014[-0.060,+0.031]; disgust:-0.054[-0.099,-0.006]; anger:-0.087!![-0.135,-0.044] | 0.103 |
| Qwen3-4B | trust_none | 0.001 | disgust:+0.005[+0.001,+0.010]; sadness:+0.004[-0.001,+0.009]; fear:+0.002[-0.003,+0.009]; anger:+0.000[-0.004,+0.004]; happiness:-0.001[-0.004,+0.000]; surprise:-0.001[-0.004,+0.000] | 0.006 |
| Qwen3-4B | unknown | 0.000 | disgust:+0.001250;anger:+0.000000;fear:+0.000000;happiness:+0.000000;sadness:+0.000000;surprise:+0.000000 | 0.001 |
| gemma-3-1b-it | trust_high | 0.459 | disgust:+0.010417;fear:+0.009167;happiness:-0.008750;sadness:-0.011667;surprise:-0.022917;anger:-0.037917 | 0.048 |
| gemma-3-1b-it | trust_low | 0.416 | anger:+0.020833;surprise:+0.012917;sadness:+0.005417;happiness:+0.002083;fear:-0.006667;disgust:-0.022083 | 0.043 |
| gemma-3-1b-it | trust_none | 0.059 | disgust:+0.010000;anger:+0.005833;sadness:+0.002500;surprise:+0.002083;happiness:+0.000000;fear:-0.006667 | 0.017 |
| gemma-3-1b-it | unknown | 0.066 | anger:+0.011250;surprise:+0.007917;happiness:+0.006667;fear:+0.004167;sadness:+0.003750;disgust:+0.001667 | 0.010 |
| gemma-3-270m-it | trust_high | 0.280 | sadness:+0.064167;anger:+0.054583;surprise:+0.050833;happiness:+0.047083;fear:+0.042083;disgust:+0.039583 | 0.025 |
| gemma-3-270m-it | trust_low | 0.341 | fear:-0.017917;happiness:-0.023333;surprise:-0.026667;anger:-0.027083;disgust:-0.033750;sadness:-0.049167 | 0.031 |
| gemma-3-270m-it | trust_none | 0.263 | disgust:+0.003333;happiness:-0.005833;sadness:-0.007500;surprise:-0.012500;anger:-0.015000;fear:-0.022500 | 0.026 |
| gemma-3-270m-it | unknown | 0.116 | fear:-0.001667;sadness:-0.007500;disgust:-0.009167;surprise:-0.011667;anger:-0.012500;happiness:-0.017917 | 0.016 |
| gemma-3-4b-it | trust_high | 0.530 | sadness:+0.008333;anger:+0.006667;surprise:+0.001250;disgust:-0.000833;happiness:-0.001250;fear:-0.010833 | 0.019 |
| gemma-3-4b-it | trust_low | 0.453 | fear:+0.004167;happiness:+0.002083;surprise:-0.001667;disgust:-0.004167;sadness:-0.005417;anger:-0.005833 | 0.010 |
| gemma-3-4b-it | trust_none | 0.016 | fear:+0.005000;disgust:+0.004583;surprise:+0.000417;happiness:-0.002917;anger:-0.003750;sadness:-0.004167 | 0.009 |
| gemma-3-4b-it | unknown | 0.001 | anger:+0.002917;happiness:+0.002083;fear:+0.001667;sadness:+0.001250;disgust:+0.000417;surprise:+0.000000 | 0.003 |

#### Item Change vs Neutral (paired by item_id, ignores intensity)
| model | change rates (emotion:%, n) |
|---|---|
| Llama-3.2-1B-Instruct | disgust:56.7% (n=623); anger:52.6% (n=626); fear:51.6% (n=607); happiness:51.1% (n=603); surprise:49.5% (n=469); sadness:44.0% (n=232) |
| Llama-3.2-3B-Instruct | disgust:89.9% (n=739); anger:71.2% (n=739); surprise:58.5% (n=728); fear:55.4% (n=734); sadness:34.9% (n=736); happiness:33.2% (n=740) |
| Phi-3.5-mini-instruct | happiness:35.3% (n=799); surprise:34.0% (n=799); sadness:33.4% (n=799); disgust:32.7% (n=799); fear:31.9% (n=799); anger:31.0% (n=799) |
| Phi-4-mini-instruct | happiness:39.2% (n=799); disgust:38.4% (n=799); fear:35.3% (n=799); anger:35.1% (n=798); surprise:33.2% (n=799); sadness:30.9% (n=799) |
| Qwen2.5-0.5B-Instruct | sadness:65.5% (n=653); disgust:61.8% (n=639); surprise:59.7% (n=636); happiness:58.7% (n=571); anger:55.7% (n=654); fear:50.8% (n=657) |
| Qwen2.5-1.5B-Instruct | happiness:53.6% (n=787); anger:52.2% (n=786); disgust:51.4% (n=786); sadness:50.9% (n=788); surprise:50.4% (n=786); fear:48.3% (n=787) |
| Qwen2.5-3B-Instruct | disgust:54.5% (n=798); happiness:52.1% (n=797); sadness:49.4% (n=798); surprise:49.0% (n=798); anger:47.6% (n=798); fear:47.4% (n=798) |
| Qwen3-0.6B | anger:41.8% (n=799); surprise:39.5% (n=799); happiness:34.3% (n=798); sadness:34.2% (n=799); fear:32.0% (n=797); disgust:31.7% (n=798) |
| Qwen3-1.7B | surprise:39.0% (n=800); anger:38.8% (n=800); disgust:38.4% (n=800); happiness:37.0% (n=800); fear:36.5% (n=800); sadness:36.5% (n=800) |
| Qwen3-4B | sadness:46.9% (n=800); fear:46.6% (n=800); anger:45.8% (n=800); disgust:45.4% (n=800); happiness:43.6% (n=800); surprise:43.1% (n=800) |
| gemma-3-1b-it | fear:48.4% (n=730); happiness:48.3% (n=735); sadness:48.1% (n=736); anger:47.5% (n=730); surprise:46.2% (n=731); disgust:45.8% (n=736) |
| gemma-3-270m-it | anger:61.1% (n=697); sadness:60.0% (n=695); disgust:59.7% (n=695); happiness:59.5% (n=697); fear:58.8% (n=690); surprise:58.4% (n=697) |
| gemma-3-4b-it | disgust:44.9% (n=798); anger:44.0% (n=797); happiness:43.9% (n=797); fear:43.4% (n=798); surprise:41.9% (n=798); sadness:39.9% (n=797) |

### Ultimatum_Game_Proposer
| model | behavior_label | neutral | all emotion deltas (Δ vs neutral) | range |
|---|---|---:|---|---:|
| Llama-3.2-1B-Instruct | offer_high | 0.170 | anger:+0.067917;disgust:+0.038333;sadness:-0.011250;surprise:-0.021667;happiness:-0.023750;fear:-0.051250 | 0.119 |
| Llama-3.2-1B-Instruct | offer_low | 0.570 | happiness:-0.013333;fear:-0.142083;disgust:-0.165417;anger:-0.207500;surprise:-0.215833;sadness:-0.342500 | 0.329 |
| Llama-3.2-1B-Instruct | offer_medium | 0.250 | anger:+0.034583;fear:-0.000833;disgust:-0.030833;happiness:-0.036667;surprise:-0.044583;sadness:-0.090000 | 0.125 |
| Llama-3.2-1B-Instruct | unknown | 0.010 | sadness:+0.443750;surprise:+0.282083;fear:+0.194167;disgust:+0.157917;anger:+0.105000;happiness:+0.073750 | 0.370 |
| Llama-3.2-3B-Instruct | offer_high | 0.593 |  | 0.226 |
| Llama-3.2-3B-Instruct | offer_low | 0.249 |  | 0.138 |
| Llama-3.2-3B-Instruct | offer_medium | 0.158 |  | 0.344 |
| Llama-3.2-3B-Instruct | unknown | 0.001 | surprise:+0.030000;anger:+0.026250;sadness:+0.020000;disgust:+0.010625;fear:+0.000417;happiness:-0.001250 | 0.031 |
| Phi-3.5-mini-instruct | offer_high | 0.864 | sadness:+0.034[+0.008,+0.064]; disgust:+0.016[-0.016,+0.048]; surprise:+0.016[-0.016,+0.048]; anger:+0.015[-0.020,+0.044]; happiness:+0.011[-0.027,+0.050]; fear:+0.005[-0.040,+0.045] | 0.013 |
| Phi-3.5-mini-instruct | offer_low | 0.043 | fear:+0.010[-0.005,+0.030]; disgust:+0.000[-0.022,+0.022]; anger:-0.010[-0.025,+0.010]; surprise:-0.011[-0.032,+0.011]; happiness:-0.011[-0.038,+0.011]; sadness:-0.015[-0.038,+0.000] | 0.017 |
| Phi-3.5-mini-instruct | offer_medium | 0.084 | happiness:+0.000[-0.034,+0.031]; anger:-0.005[-0.044,+0.030]; surprise:-0.005[-0.043,+0.027]; fear:-0.015[-0.054,+0.025]; disgust:-0.016[-0.059,+0.016]; sadness:-0.019[-0.045,+0.004] | 0.008 |
| Phi-3.5-mini-instruct | unknown | 0.010 | disgust:-0.003333;sadness:-0.003750;fear:-0.004167;happiness:-0.005833;anger:-0.006250;surprise:-0.006667 | 0.003 |
| Phi-4-mini-instruct | offer_high | 0.845 |  | 0.466 |
| Phi-4-mini-instruct | offer_low | 0.046 |  | 0.024 |
| Phi-4-mini-instruct | offer_medium | 0.109 |  | 0.108 |
| Phi-4-mini-instruct | unknown | 0.000 | sadness:+0.535417;disgust:+0.085417;anger:+0.025000;fear:+0.000000;happiness:+0.000000;surprise:+0.000000 | 0.535 |
| Qwen2.5-0.5B-Instruct | offer_high | 0.279 | sadness:+0.181!!![+0.107,+0.246]; anger:+0.127!![+0.058,+0.188]; fear:+0.120!![+0.052,+0.173]; surprise:+0.078[+0.011,+0.143]; disgust:-0.069[-0.130,+0.000]; happiness:-0.071[-0.126,-0.011] | 0.133 |
| Qwen2.5-0.5B-Instruct | offer_low | 0.401 | happiness:+0.049[-0.023,+0.117]; disgust:+0.039[-0.027,+0.108]; surprise:+0.011[-0.057,+0.070]; anger:-0.039[-0.107,+0.039]; fear:-0.063[-0.123,-0.003]; sadness:-0.110![-0.178,-0.039] | 0.129 |
| Qwen2.5-0.5B-Instruct | offer_medium | 0.314 | disgust:+0.030[-0.042,+0.084]; happiness:+0.023[-0.051,+0.094]; fear:-0.058[-0.115,+0.000]; sadness:-0.071[-0.142,-0.006]; anger:-0.088![-0.162,-0.029]; surprise:-0.089![-0.148,-0.030] | 0.110 |
| Qwen2.5-0.5B-Instruct | unknown | 0.006 | sadness:+0.107500;surprise:+0.039167;anger:+0.026250;fear:+0.020417;disgust:+0.017917;happiness:-0.000417 | 0.108 |
| Qwen2.5-1.5B-Instruct | offer_high | 0.512 | disgust:+0.172[-0.069,+0.379]; happiness:-0.025[-0.175,+0.113]; anger:-0.040[-0.170,+0.080]; fear:-0.053[-0.200,+0.107]; sadness:-0.061[-0.183,+0.052]; surprise:-0.067[-0.187,+0.080] | 0.117 |
| Qwen2.5-1.5B-Instruct | offer_low | 0.287 | fear:+0.080[-0.053,+0.213]; happiness:+0.062[-0.062,+0.188]; surprise:+0.040[-0.093,+0.160]; anger:+0.020[-0.090,+0.130]; sadness:+0.009[-0.104,+0.122]; disgust:-0.069[-0.276,+0.103] | 0.139 |
| Qwen2.5-1.5B-Instruct | offer_medium | 0.200 | sadness:+0.052[-0.052,+0.148]; surprise:+0.027[-0.093,+0.133]; anger:+0.020[-0.100,+0.120]; fear:-0.027[-0.133,+0.080]; happiness:-0.038[-0.150,+0.075]; disgust:-0.103[-0.276,+0.069] | 0.050 |
| Qwen2.5-1.5B-Instruct | unknown | 0.000 | disgust:+0.006667;sadness:+0.005417;anger:+0.004167;fear:+0.003750;happiness:+0.001250;surprise:+0.001250 | 0.005 |
| Qwen2.5-3B-Instruct | offer_high | 0.456 | sadness:+0.085![+0.025,+0.137]; fear:-0.002[-0.053,+0.053]; happiness:-0.019[-0.079,+0.037]; anger:-0.038[-0.101,+0.019]; surprise:-0.049[-0.103,+0.004]; disgust:-0.074[-0.132,-0.014] | 0.141 |
| Qwen2.5-3B-Instruct | offer_low | 0.328 | surprise:+0.054[+0.007,+0.099]; disgust:+0.037[-0.016,+0.082]; happiness:+0.033[-0.016,+0.078]; anger:+0.013[-0.044,+0.065]; fear:+0.009[-0.047,+0.060]; sadness:-0.056[-0.098,-0.010] | 0.141 |
| Qwen2.5-3B-Instruct | offer_medium | 0.216 | disgust:+0.037[-0.014,+0.087]; anger:+0.025[-0.025,+0.076]; surprise:-0.004[-0.054,+0.047]; fear:-0.006[-0.047,+0.043]; happiness:-0.014[-0.068,+0.037]; sadness:-0.029[-0.077,+0.015] | 0.037 |
| Qwen3-0.6B | offer_high | 0.799 | surprise:+0.036[-0.048,+0.119]; sadness:+0.007[-0.072,+0.087]; disgust:+0.000[-0.059,+0.063]; fear:+0.000[-0.118,+0.118]; happiness:-0.035[-0.095,+0.020]; anger:-0.107[-0.268,+0.036] | 0.058 |
| Qwen3-0.6B | offer_low | 0.100 | fear:+0.078[-0.059,+0.196]; happiness:+0.005[-0.035,+0.045]; anger:+0.000[-0.125,+0.107]; sadness:+0.000[-0.058,+0.058]; surprise:-0.012[-0.095,+0.060]; disgust:-0.015[-0.059,+0.029] | 0.050 |
| Qwen3-0.6B | offer_medium | 0.101 | anger:+0.107[+0.000,+0.250]; happiness:+0.030[-0.020,+0.085]; disgust:+0.015[-0.044,+0.063]; sadness:-0.007[-0.080,+0.065]; surprise:-0.024[-0.119,+0.071]; fear:-0.078[-0.196,+0.039] | 0.024 |
| Qwen3-0.6B | unknown | 0.000 | disgust:+0.002083;anger:+0.000000;fear:+0.000000;happiness:+0.000000;sadness:+0.000000;surprise:+0.000000 | 0.002 |
| Qwen3-1.7B | offer_high | 0.630 | anger:+0.000[+0.000,+0.000]; disgust:+0.000[+0.000,+0.000]; fear:+0.000[+0.000,+0.000]; happiness:+0.000[+0.000,+0.000]; sadness:+0.000[+0.000,+0.000]; surprise:+0.000[+0.000,+0.000] | 0.054 |
| Qwen3-1.7B | offer_low | 0.256 | anger:+0.000[+0.000,+0.000]; disgust:+0.000[+0.000,+0.000]; fear:+0.000[+0.000,+0.000]; happiness:+0.000[+0.000,+0.000]; sadness:+0.000[+0.000,+0.000]; surprise:+0.000[+0.000,+0.000] | 0.048 |
| Qwen3-1.7B | offer_medium | 0.114 | anger:+0.000[+0.000,+0.000]; disgust:+0.000[+0.000,+0.000]; fear:+0.000[+0.000,+0.000]; happiness:+0.000[+0.000,+0.000]; sadness:+0.000[+0.000,+0.000]; surprise:+0.000[+0.000,+0.000] | 0.018 |
| Qwen3-4B | offer_high | 0.709 | sadness:+0.004[-0.036,+0.037]; surprise:-0.008[-0.044,+0.030]; fear:-0.014[-0.051,+0.021]; disgust:-0.037[-0.078,+0.001]; happiness:-0.050[-0.093,-0.012]; anger:-0.151!!![-0.195,-0.106] | 0.130 |
| Qwen3-4B | offer_low | 0.102 | anger:+0.089!!![+0.054,+0.121]; disgust:+0.041![+0.011,+0.066]; happiness:+0.035[+0.008,+0.061]; fear:+0.027[+0.001,+0.055]; surprise:+0.020[-0.011,+0.050]; sadness:-0.004[-0.031,+0.020] | 0.068 |
| Qwen3-4B | offer_medium | 0.189 | anger:+0.062!![+0.025,+0.102]; happiness:+0.016[-0.021,+0.056]; sadness:+0.000[-0.036,+0.035]; disgust:-0.004[-0.037,+0.032]; surprise:-0.012[-0.046,+0.022]; fear:-0.013[-0.050,+0.024] | 0.062 |
| gemma-3-1b-it | offer_high | 0.374 | anger:+0.062083;fear:+0.040833;disgust:+0.035833;happiness:+0.016667;sadness:+0.012083;surprise:+0.009167 | 0.053 |
| gemma-3-1b-it | offer_low | 0.328 | disgust:-0.009167;sadness:-0.015833;happiness:-0.016250;fear:-0.018750;surprise:-0.020833;anger:-0.062500 | 0.053 |
| gemma-3-1b-it | offer_medium | 0.296 | surprise:+0.012917;sadness:+0.003333;happiness:+0.000833;anger:-0.002083;fear:-0.024167;disgust:-0.028333 | 0.041 |
| gemma-3-1b-it | unknown | 0.003 | anger:+0.002500;fear:+0.002083;disgust:+0.001667;surprise:+0.001250;sadness:+0.000417;happiness:-0.001250 | 0.004 |
| gemma-3-270m-it | offer_high | 0.228 | anger:+0.019167;disgust:+0.017083;sadness:+0.013333;surprise:-0.000417;fear:-0.009583;happiness:-0.015833 | 0.035 |
| gemma-3-270m-it | offer_low | 0.281 | surprise:-0.007917;anger:-0.012083;fear:-0.019583;happiness:-0.024583;sadness:-0.029167;disgust:-0.032083 | 0.024 |
| gemma-3-270m-it | offer_medium | 0.375 | fear:+0.026667;happiness:+0.019583;sadness:+0.012917;surprise:+0.010417;disgust:+0.007500;anger:-0.013750 | 0.040 |
| gemma-3-270m-it | unknown | 0.116 | happiness:+0.020833;disgust:+0.007500;anger:+0.006667;sadness:+0.002917;fear:+0.002500;surprise:-0.002083 | 0.023 |
| gemma-3-4b-it | offer_high | 0.301 | anger:+0.080417;disgust:+0.073750;happiness:+0.067083;fear:+0.057917;surprise:+0.040417;sadness:+0.035417 | 0.045 |
| gemma-3-4b-it | offer_low | 0.341 | fear:-0.028750;sadness:-0.036667;anger:-0.053750;surprise:-0.063750;disgust:-0.064583;happiness:-0.065000 | 0.036 |
| gemma-3-4b-it | offer_medium | 0.357 | surprise:+0.023333;sadness:+0.001250;happiness:-0.002083;disgust:-0.009167;anger:-0.026667;fear:-0.029167 | 0.052 |

#### Item Change vs Neutral (paired by item_id, ignores intensity)
| model | change rates (emotion:%, n) |
|---|---|
| Llama-3.2-1B-Instruct | anger:63.0% (n=791); sadness:62.4% (n=752); surprise:55.7% (n=788); disgust:55.0% (n=787); happiness:52.1% (n=791); fear:51.0% (n=779) |
| Llama-3.2-3B-Instruct | fear:65.6% (n=799); surprise:61.1% (n=799); sadness:60.2% (n=799); anger:56.2% (n=799); disgust:53.9% (n=799); happiness:48.6% (n=799) |
| Phi-3.5-mini-instruct | disgust:11.4% (n=792); fear:9.9% (n=791); surprise:9.7% (n=792); anger:9.2% (n=791); sadness:9.1% (n=792); happiness:9.0% (n=790) |
| Phi-4-mini-instruct | disgust:20.6% (n=798); surprise:18.1% (n=800); sadness:16.4% (n=697); anger:16.2% (n=800); happiness:12.1% (n=800); fear:11.6% (n=800) |
| Qwen2.5-0.5B-Instruct | sadness:66.9% (n=791); surprise:64.1% (n=794); anger:63.6% (n=794); disgust:63.3% (n=792); fear:63.2% (n=793); happiness:58.4% (n=795) |
| Qwen2.5-1.5B-Instruct | fear:57.5% (n=800); happiness:56.1% (n=800); sadness:55.4% (n=799); disgust:53.9% (n=800); anger:51.2% (n=800); surprise:50.6% (n=800) |
| Qwen2.5-3B-Instruct | disgust:59.2% (n=800); happiness:58.4% (n=800); anger:57.0% (n=800); surprise:55.1% (n=800); fear:54.9% (n=800); sadness:54.8% (n=800) |
| Qwen3-0.6B | anger:27.9% (n=800); sadness:27.6% (n=800); surprise:26.9% (n=800); disgust:25.0% (n=800); happiness:24.2% (n=800); fear:24.1% (n=800) |
| Qwen3-1.7B | surprise:39.8% (n=800); happiness:39.1% (n=800); fear:39.0% (n=800); disgust:38.4% (n=800); anger:37.6% (n=800); sadness:36.9% (n=800) |
| Qwen3-4B | anger:42.2% (n=800); happiness:38.4% (n=800); disgust:37.5% (n=800); surprise:35.4% (n=800); sadness:33.0% (n=800); fear:32.8% (n=800) |
| gemma-3-1b-it | sadness:59.0% (n=797); disgust:57.4% (n=798); happiness:56.0% (n=798); anger:55.8% (n=796); fear:55.6% (n=798); surprise:52.8% (n=798) |
| gemma-3-270m-it | disgust:59.0% (n=670); anger:57.8% (n=671); happiness:57.6% (n=660); surprise:56.0% (n=668); sadness:54.9% (n=670); fear:54.0% (n=669) |
| gemma-3-4b-it | surprise:62.0% (n=800); fear:61.6% (n=800); sadness:61.4% (n=800); disgust:61.1% (n=800); anger:59.9% (n=800); happiness:59.6% (n=800) |

### Ultimatum_Game_Responder
| model | behavior_label | neutral | all emotion deltas (Δ vs neutral) | range |
|---|---|---:|---|---:|
| Llama-3.2-1B-Instruct | accept | 0.656 | happiness:+0.237500;surprise:-0.100000;sadness:-0.312083;disgust:-0.428750;anger:-0.500833;fear:-0.548750 | 0.786 |
| Llama-3.2-1B-Instruct | reject | 0.336 | fear:+0.550417;anger:+0.483333;disgust:+0.429583;sadness:-0.090833;surprise:-0.162917;happiness:-0.278750 | 0.829 |
| Llama-3.2-1B-Instruct | unknown | 0.007 | sadness:+0.402917;surprise:+0.262917;anger:+0.067500;happiness:+0.065625;disgust:-0.000833;fear:-0.001667 | 0.405 |
| Llama-3.2-3B-Instruct | accept | 0.931 | disgust:+0.000[-0.429,+0.429]; sadness:-0.080[-0.320,+0.160]; surprise:-0.167[-0.375,+0.000]; fear:-0.211[-0.474,+0.053]; anger:-0.308[-0.538,-0.115] | 0.687 |
| Llama-3.2-3B-Instruct | reject | 0.069 | anger:+0.308[+0.115,+0.538]; fear:+0.211[-0.053,+0.474]; surprise:+0.167[-0.042,+0.375]; sadness:+0.080[-0.160,+0.280]; disgust:+0.000[-0.429,+0.429] | 0.682 |
| Llama-3.2-3B-Instruct | unknown | 0.000 | sadness:+0.007500;anger:+0.005000;surprise:+0.003750;fear:+0.002500;disgust:+0.001875;happiness:+0.000000 | 0.007 |
| Phi-3.5-mini-instruct | accept | 0.976 | happiness:+0.012![+0.003,+0.022]; surprise:+0.005[-0.003,+0.012]; sadness:+0.002[-0.006,+0.009]; fear:-0.029!!![-0.045,-0.016]; disgust:-0.034!!![-0.050,-0.021]; anger:-0.083!!![-0.104,-0.062] | 0.142 |
| Phi-3.5-mini-instruct | reject | 0.018 | anger:+0.083!!![+0.060,+0.104]; disgust:+0.034!!![+0.019,+0.050]; fear:+0.029!!![+0.016,+0.044]; sadness:-0.002[-0.009,+0.006]; surprise:-0.005[-0.012,+0.003]; happiness:-0.012![-0.022,-0.003] | 0.127 |
| Phi-3.5-mini-instruct | unknown | 0.006 | disgust:+0.018333;anger:+0.008750;fear:+0.006250;surprise:-0.002917;sadness:-0.003333;happiness:-0.006250 | 0.025 |
| Phi-4-mini-instruct | accept | 0.715 | happiness:+0.206!!![+0.156,+0.266]; fear:+0.075[+0.010,+0.136]; anger:+0.000[+0.000,+0.000]; surprise:+0.000[+0.000,+0.000]; sadness:-0.235[-0.412,-0.059]; disgust:-0.479!!![-0.625,-0.354] | 0.760 |
| Phi-4-mini-instruct | reject | 0.186 | disgust:+0.479!!![+0.333,+0.625]; sadness:+0.235[+0.059,+0.412]; anger:+0.000[+0.000,+0.000]; surprise:+0.000[+0.000,+0.000]; fear:-0.075[-0.141,-0.015]; happiness:-0.206!!![-0.266,-0.156] | 0.369 |
| Phi-4-mini-instruct | unknown | 0.099 | sadness:+0.547917;anger:+0.309167;disgust:+0.287083;fear:-0.052917;surprise:-0.058333;happiness:-0.092500 | 0.640 |
| Qwen2.5-0.5B-Instruct | accept | 0.637 | happiness:+0.317!!![+0.279,+0.357]; surprise:-0.110!!![-0.156,-0.063]; fear:-0.219!!![-0.267,-0.173]; disgust:-0.248!!![-0.291,-0.198]; sadness:-0.495!!![-0.541,-0.450]; anger:-0.585!!![-0.625,-0.544] | 0.838 |
| Qwen2.5-0.5B-Instruct | reject | 0.362 | anger:+0.585!!![+0.542,+0.625]; sadness:+0.495!!![+0.450,+0.541]; disgust:+0.248!!![+0.197,+0.291]; fear:+0.219!!![+0.173,+0.265]; surprise:+0.110!!![+0.061,+0.156]; happiness:-0.317!!![-0.357,-0.279] | 0.838 |
| Qwen2.5-0.5B-Instruct | unknown | 0.000 | sadness:+0.024167;surprise:+0.003125;fear:+0.001250;happiness:+0.001250;anger:+0.000000;disgust:+0.000000 | 0.024 |
| Qwen2.5-1.5B-Instruct | accept | 0.605 | happiness:+0.045[-0.057,+0.159]; surprise:+0.030[-0.075,+0.149]; disgust:+0.000[-0.200,+0.200]; anger:-0.083[-0.250,+0.083]; sadness:-0.101[-0.228,+0.013]; fear:-0.175[-0.333,-0.018] | 0.203 |
| Qwen2.5-1.5B-Instruct | reject | 0.394 | fear:+0.175[+0.000,+0.333]; sadness:+0.101[-0.013,+0.215]; anger:+0.083[-0.083,+0.250]; disgust:+0.000[-0.200,+0.200]; surprise:-0.030[-0.164,+0.075]; happiness:-0.045[-0.159,+0.057] | 0.203 |
| Qwen2.5-1.5B-Instruct | unknown | 0.001 | sadness:+0.002500;fear:+0.000833;surprise:+0.000625;disgust:+0.000417;anger:+0.000000;happiness:-0.001250 | 0.004 |
| Qwen2.5-3B-Instruct | accept | 0.870 | happiness:+0.066!!![+0.041,+0.091]; surprise:+0.000[-0.031,+0.029]; sadness:-0.038![-0.070,-0.003]; disgust:-0.072!!![-0.102,-0.037]; fear:-0.105!!![-0.140,-0.072]; anger:-0.281!!![-0.320,-0.243] | 0.250 |
| Qwen2.5-3B-Instruct | reject | 0.129 | anger:+0.281!!![+0.243,+0.320]; fear:+0.105!!![+0.072,+0.140]; disgust:+0.072!!![+0.036,+0.101]; sadness:+0.038![+0.000,+0.069]; surprise:+0.000[-0.029,+0.031]; happiness:-0.066!!![-0.093,-0.041] | 0.251 |
| Qwen2.5-3B-Instruct | unknown | 0.001 | sadness:+0.003333;happiness:+0.002500;disgust:+0.000000;fear:+0.000000;anger:-0.001250;surprise:-0.001250 | 0.005 |
| Qwen3-0.6B | accept | 0.816 | happiness:+0.180!!![+0.138,+0.225]; sadness:+0.119!![+0.045,+0.187]; surprise:+0.055[+0.000,+0.110]; fear:-0.094![-0.164,-0.023]; disgust:-0.188!!![-0.232,-0.144]; anger:-0.191!![-0.298,-0.085] | 0.378 |
| Qwen3-0.6B | reject | 0.184 | anger:+0.191!![+0.085,+0.298]; disgust:+0.188!!![+0.144,+0.232]; fear:+0.094![+0.023,+0.164]; surprise:-0.055[-0.110,-0.008]; sadness:-0.119!![-0.187,-0.060]; happiness:-0.180!!![-0.225,-0.138] | 0.378 |
| Qwen3-0.6B | unknown | 0.000 | disgust:+0.001250;anger:+0.000000;fear:+0.000000;happiness:+0.000000;sadness:+0.000000;surprise:+0.000000 | 0.001 |
| Qwen3-1.7B | accept | 0.963 | happiness:+0.004[+0.000,+0.011]; fear:+0.000[-0.006,+0.006]; disgust:-0.004[-0.010,+0.000]; sadness:-0.004[-0.010,+0.000]; surprise:-0.004[-0.015,+0.004]; anger:-0.011[-0.023,-0.004] | 0.022 |
| Qwen3-1.7B | reject | 0.037 | anger:+0.011[+0.002,+0.021]; surprise:+0.004[-0.004,+0.013]; sadness:+0.004[+0.000,+0.010]; disgust:+0.004[+0.000,+0.010]; fear:+0.000[-0.006,+0.006]; happiness:-0.004[-0.011,+0.000] | 0.022 |
| Qwen3-4B | accept | 0.943 | happiness:+0.030!![+0.013,+0.049]; sadness:+0.014[-0.001,+0.029]; anger:+0.009[-0.006,+0.023]; fear:-0.005[-0.023,+0.010]; surprise:-0.021[-0.040,-0.005]; disgust:-0.035!![-0.054,-0.016] | 0.042 |
| Qwen3-4B | reject | 0.058 | disgust:+0.035!![+0.016,+0.053]; surprise:+0.021[+0.004,+0.039]; fear:+0.005[-0.010,+0.021]; anger:-0.009[-0.024,+0.006]; sadness:-0.014[-0.029,+0.001]; happiness:-0.030!![-0.049,-0.013] | 0.042 |
| gemma-3-1b-it | accept | 0.641 | fear:+0.012917;happiness:-0.008333;disgust:-0.017083;anger:-0.023333;surprise:-0.031250;sadness:-0.055833 | 0.069 |
| gemma-3-1b-it | reject | 0.359 | sadness:+0.055833;surprise:+0.030833;anger:+0.023333;disgust:+0.017083;happiness:+0.008333;fear:-0.012917 | 0.069 |
| gemma-3-1b-it | unknown | 0.000 | surprise:+0.001250;anger:+0.000000;disgust:+0.000000;fear:+0.000000;happiness:+0.000000;sadness:+0.000000 | 0.001 |
| gemma-3-270m-it | accept | 0.412 | surprise:-0.010417;anger:-0.013333;sadness:-0.020000;fear:-0.022500;disgust:-0.024583;happiness:-0.034167 | 0.024 |
| gemma-3-270m-it | reject | 0.588 | happiness:+0.034167;disgust:+0.024583;fear:+0.022500;sadness:+0.020000;anger:+0.013333;surprise:+0.010417 | 0.024 |
| gemma-3-4b-it | accept | 0.723 | surprise:+0.024167;happiness:+0.017500;sadness:+0.015000;fear:+0.010833;disgust:+0.004583;anger:-0.004167 | 0.028 |
| gemma-3-4b-it | reject | 0.278 | anger:+0.004167;disgust:-0.004583;fear:-0.010833;sadness:-0.015000;happiness:-0.017500;surprise:-0.024167 | 0.028 |

#### Item Change vs Neutral (paired by item_id, ignores intensity)
| model | change rates (emotion:%, n) |
|---|---|
| Llama-3.2-1B-Instruct | fear:63.7% (n=794); disgust:60.7% (n=794); anger:55.3% (n=794); surprise:43.1% (n=786); sadness:39.1% (n=727); happiness:35.4% (n=794) |
| Llama-3.2-3B-Instruct | fear:83.1% (n=800); anger:79.1% (n=800); surprise:71.1% (n=800); disgust:55.5% (n=800); sadness:44.9% (n=800); happiness:6.9% (n=800) |
| Phi-3.5-mini-instruct | anger:19.4% (n=793); disgust:10.4% (n=795); fear:9.2% (n=795); sadness:2.8% (n=795); surprise:2.1% (n=795); happiness:1.5% (n=795) |
| Phi-4-mini-instruct | disgust:55.3% (n=636); anger:43.5% (n=626); sadness:35.2% (n=506); happiness:20.5% (n=721); surprise:17.9% (n=716); fear:17.7% (n=719) |
| Qwen2.5-0.5B-Instruct | anger:62.0% (n=800); sadness:54.9% (n=800); fear:50.5% (n=800); disgust:47.4% (n=800); surprise:39.4% (n=800); happiness:38.5% (n=800) |
| Qwen2.5-1.5B-Instruct | disgust:49.4% (n=799); fear:49.4% (n=799); anger:45.2% (n=799); sadness:44.7% (n=799); surprise:44.6% (n=799); happiness:41.4% (n=799) |
| Qwen2.5-3B-Instruct | anger:37.4% (n=799); fear:26.5% (n=799); disgust:22.9% (n=799); sadness:19.8% (n=799); surprise:19.6% (n=799); happiness:16.5% (n=799) |
| Qwen3-0.6B | anger:35.8% (n=800); disgust:21.4% (n=800); fear:18.1% (n=800); happiness:16.8% (n=800); sadness:15.9% (n=800); surprise:9.5% (n=800) |
| Qwen3-1.7B | anger:5.0% (n=800); disgust:4.4% (n=800); surprise:4.4% (n=800); fear:4.0% (n=800); sadness:3.5% (n=800); happiness:3.0% (n=800) |
| Qwen3-4B | surprise:8.6% (n=800); disgust:8.4% (n=800); fear:6.5% (n=800); happiness:6.4% (n=800); sadness:5.4% (n=800); anger:5.0% (n=800) |
| gemma-3-1b-it | sadness:33.4% (n=800); anger:33.0% (n=800); disgust:32.6% (n=800); happiness:32.1% (n=800); fear:31.1% (n=800); surprise:29.6% (n=800) |
| gemma-3-270m-it | happiness:23.4% (n=800); surprise:23.2% (n=800); anger:22.5% (n=800); fear:22.2% (n=800); sadness:21.6% (n=800); disgust:20.8% (n=800) |
| gemma-3-4b-it | surprise:23.9% (n=800); sadness:23.6% (n=800); anger:23.1% (n=800); happiness:23.0% (n=800); fear:21.9% (n=800); disgust:19.0% (n=800) |

## Option Intensity Sensitivity (Top 20 by delta_range_across_intensity)
| game_setting | model | emotion | option_id | best (intensity, Δ) | worst (intensity, Δ) | range |
|---|---|---|---|---|---|---:|
| Trust_Game_Trustor | Llama-3.2-3B-Instruct | sadness | -1 | 1.2 (+0.911) | 0.6 (-0.062) | 0.974 |
| Trust_Game_Trustee | Llama-3.2-3B-Instruct | sadness | -1 | 1.2 (+0.865) | 0.6 (-0.077) | 0.943 |
| Trust_Game_Trustee | Llama-3.2-1B-Instruct | anger | -1 | 1.2 (+0.853) | 0.6 (+0.005) | 0.848 |
| Trust_Game_Trustee | Llama-3.2-3B-Instruct | surprise | -1 | 1.2 (+0.780) | 0.6 (-0.062) | 0.843 |
| Trust_Game_Trustor | Llama-3.2-3B-Instruct | surprise | -1 | 1.2 (+0.799) | 0.6 (-0.033) | 0.831 |
| Trust_Game_Trustor | Phi-4-mini-instruct | sadness | -1 | 1.2 (+0.824) | 0.6 (-0.001) | 0.825 |
| Ultimatum_Game_Proposer | Llama-3.2-1B-Instruct | surprise | -1 | 1.2 (+0.807) | 0.6 (+0.013) | 0.795 |
| Trust_Game_Trustee | Llama-3.2-1B-Instruct | happiness | -1 | 1.2 (+0.805) | 0.6 (+0.037) | 0.767 |
| Ultimatum_Game_Proposer | Phi-4-mini-instruct | sadness | -1 | 1.2 (+0.909) | 0.6 (+0.156) | 0.752 |
| Trust_Game_Trustor | Llama-3.2-1B-Instruct | happiness | -1 | 1.2 (+0.801) | 0.6 (+0.071) | 0.730 |
| Ultimatum_Game_Proposer | Llama-3.2-1B-Instruct | sadness | -1 | 1.2 (+0.830) | 0.6 (+0.122) | 0.708 |
| Trust_Game_Trustor | Llama-3.2-1B-Instruct | anger | -1 | 1.2 (+0.734) | 0.6 (+0.034) | 0.700 |
| Ultimatum_Game_Responder | Llama-3.2-1B-Instruct | surprise | -1 | 1.2 (+0.659) | 0.6 (+0.030) | 0.629 |
| Trust_Game_Trustee | Phi-4-mini-instruct | sadness | -1 | 1.2 (+0.912) | 0.6 (+0.292) | 0.620 |
| Trust_Game_Trustor | Llama-3.2-1B-Instruct | surprise | -1 | 1.2 (+0.846) | 0.6 (+0.249) | 0.598 |
| Trust_Game_Trustee | Llama-3.2-1B-Instruct | anger | 1 | 0.6 (-0.009) | 1.2 (-0.573) | 0.564 |
| Ultimatum_Game_Proposer | Llama-3.2-1B-Instruct | surprise | 1 | 0.6 (-0.083) | 1.2 (-0.603) | 0.520 |
| Trust_Game_Trustee | Llama-3.2-1B-Instruct | surprise | -1 | 1.2 (+0.899) | 0.6 (+0.386) | 0.512 |
| Trust_Game_Trustor | Llama-3.2-1B-Instruct | anger | 1 | 0.6 (-0.021) | 1.2 (-0.522) | 0.501 |
| Trust_Game_Trustor | Llama-3.2-3B-Instruct | fear | -1 | 1.2 (+0.444) | 0.6 (-0.055) | 0.499 |

## Behavior Intensity Sensitivity (Top 20 by delta_range_across_intensity)
| game_setting | model | emotion | behavior_label | best (intensity, Δ) | worst (intensity, Δ) | range |
|---|---|---|---|---|---|---:|
| Trust_Game_Trustor | Llama-3.2-3B-Instruct | sadness | unknown | 1.2 (+0.911) | 0.6 (-0.062) | 0.974 |
| Trust_Game_Trustee | Llama-3.2-3B-Instruct | sadness | unknown | 1.2 (+0.865) | 0.6 (-0.077) | 0.943 |
| Trust_Game_Trustee | Llama-3.2-1B-Instruct | anger | unknown | 1.2 (+0.853) | 0.6 (+0.005) | 0.848 |
| Trust_Game_Trustee | Llama-3.2-3B-Instruct | surprise | unknown | 1.2 (+0.780) | 0.6 (-0.062) | 0.843 |
| Trust_Game_Trustor | Llama-3.2-3B-Instruct | surprise | unknown | 1.2 (+0.799) | 0.6 (-0.033) | 0.831 |
| Trust_Game_Trustor | Phi-4-mini-instruct | sadness | unknown | 1.2 (+0.824) | 0.6 (-0.001) | 0.825 |
| Ultimatum_Game_Proposer | Llama-3.2-1B-Instruct | surprise | unknown | 1.2 (+0.807) | 0.6 (+0.013) | 0.795 |
| Trust_Game_Trustee | Llama-3.2-1B-Instruct | happiness | unknown | 1.2 (+0.805) | 0.6 (+0.037) | 0.767 |
| Ultimatum_Game_Proposer | Phi-4-mini-instruct | sadness | unknown | 1.2 (+0.909) | 0.6 (+0.156) | 0.752 |
| Trust_Game_Trustor | Llama-3.2-1B-Instruct | happiness | unknown | 1.2 (+0.801) | 0.6 (+0.071) | 0.730 |
| Ultimatum_Game_Proposer | Llama-3.2-1B-Instruct | sadness | unknown | 1.2 (+0.830) | 0.6 (+0.122) | 0.708 |
| Trust_Game_Trustor | Llama-3.2-1B-Instruct | anger | unknown | 1.2 (+0.734) | 0.6 (+0.034) | 0.700 |
| Trust_Game_Trustee | Llama-3.2-3B-Instruct | sadness | return_high | 0.6 (+0.021) | 1.2 (-0.651) | 0.672 |
| Ultimatum_Game_Proposer | Phi-4-mini-instruct | sadness | offer_high | 0.6 (-0.147) | 1.2 (-0.791) | 0.644 |
| Trust_Game_Trustor | Llama-3.2-3B-Instruct | sadness | trust_high | 0.6 (-0.058) | 1.2 (-0.698) | 0.640 |
| Ultimatum_Game_Responder | Llama-3.2-1B-Instruct | surprise | unknown | 1.2 (+0.659) | 0.6 (+0.030) | 0.629 |
| Trust_Game_Trustee | Phi-4-mini-instruct | sadness | unknown | 1.2 (+0.912) | 0.6 (+0.292) | 0.620 |
| Trust_Game_Trustor | Llama-3.2-1B-Instruct | surprise | unknown | 1.2 (+0.846) | 0.6 (+0.249) | 0.598 |
| Trust_Game_Trustee | Llama-3.2-3B-Instruct | surprise | return_high | 0.6 (-0.050) | 1.2 (-0.632) | 0.583 |
| Ultimatum_Game_Responder | Llama-3.2-1B-Instruct | surprise | accept | 0.6 (+0.152) | 1.2 (-0.416) | 0.569 |

