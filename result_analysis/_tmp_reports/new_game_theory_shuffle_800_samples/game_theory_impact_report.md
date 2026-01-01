# Game-Theory Decision Impact Report (vs neutral)

## Data Used
- Root scanned: `results/new_game_theory/shuffle_800_samples`
- Input files searched: `**/summary_choice_ratio.csv`, `**/summary_behavior_ratio.csv`
- Latest run per (model, game_setting): 91
  - `results/new_game_theory/shuffle_800_samples/Llama-3.2-1B-Instruct_game_theory_Escalation_Game_20251226_144059`
  - `results/new_game_theory/shuffle_800_samples/Llama-3.2-1B-Instruct_game_theory_Prisoners_Dilemma_20251226_123432`
  - `results/new_game_theory/shuffle_800_samples/Llama-3.2-1B-Instruct_game_theory_Stag_Hunt_20251226_133517`
  - `results/new_game_theory/shuffle_800_samples/Llama-3.2-1B-Instruct_game_theory_Trust_Game_Trustee_20251226_164318`
  - `results/new_game_theory/shuffle_800_samples/Llama-3.2-1B-Instruct_game_theory_Trust_Game_Trustor_20251226_154126`
  - `results/new_game_theory/shuffle_800_samples/Llama-3.2-1B-Instruct_game_theory_Ultimatum_Game_Proposer_20251226_174549`
  - `results/new_game_theory/shuffle_800_samples/Llama-3.2-1B-Instruct_game_theory_Ultimatum_Game_Responder_20251226_184417`
  - `results/new_game_theory/shuffle_800_samples/Llama-3.2-3B-Instruct_game_theory_Escalation_Game_20251226_145210`
  - `results/new_game_theory/shuffle_800_samples/Llama-3.2-3B-Instruct_game_theory_Prisoners_Dilemma_20251227_061934`
  - `results/new_game_theory/shuffle_800_samples/Llama-3.2-3B-Instruct_game_theory_Stag_Hunt_20251226_134619`
  - `results/new_game_theory/shuffle_800_samples/Llama-3.2-3B-Instruct_game_theory_Trust_Game_Trustee_20251226_165402`
  - `results/new_game_theory/shuffle_800_samples/Llama-3.2-3B-Instruct_game_theory_Trust_Game_Trustor_20251226_155215`
  - `results/new_game_theory/shuffle_800_samples/Llama-3.2-3B-Instruct_game_theory_Ultimatum_Game_Proposer_20251226_175643`
  - `results/new_game_theory/shuffle_800_samples/Llama-3.2-3B-Instruct_game_theory_Ultimatum_Game_Responder_20251226_185523`
  - `results/new_game_theory/shuffle_800_samples/Phi-3.5-mini-instruct_game_theory_Escalation_Game_20251226_142610`
  - `results/new_game_theory/shuffle_800_samples/Phi-3.5-mini-instruct_game_theory_Prisoners_Dilemma_20251226_042552`
  - `results/new_game_theory/shuffle_800_samples/Phi-3.5-mini-instruct_game_theory_Stag_Hunt_20251226_131909`
  - `results/new_game_theory/shuffle_800_samples/Phi-3.5-mini-instruct_game_theory_Trust_Game_Trustee_20251226_162936`
  - `results/new_game_theory/shuffle_800_samples/Phi-3.5-mini-instruct_game_theory_Trust_Game_Trustor_20251226_152811`
  - `results/new_game_theory/shuffle_800_samples/Phi-3.5-mini-instruct_game_theory_Ultimatum_Game_Proposer_20251226_173223`
  - `results/new_game_theory/shuffle_800_samples/Phi-3.5-mini-instruct_game_theory_Ultimatum_Game_Responder_20251226_183217`
  - `results/new_game_theory/shuffle_800_samples/Phi-4-mini-instruct_game_theory_Escalation_Game_20251226_143553`
  - `results/new_game_theory/shuffle_800_samples/Phi-4-mini-instruct_game_theory_Prisoners_Dilemma_20251227_060707`
  - `results/new_game_theory/shuffle_800_samples/Phi-4-mini-instruct_game_theory_Stag_Hunt_20251226_132856`
  - `results/new_game_theory/shuffle_800_samples/Phi-4-mini-instruct_game_theory_Trust_Game_Trustee_20251226_163747`
  - `results/new_game_theory/shuffle_800_samples/Phi-4-mini-instruct_game_theory_Trust_Game_Trustor_20251226_153625`
  - `results/new_game_theory/shuffle_800_samples/Phi-4-mini-instruct_game_theory_Ultimatum_Game_Proposer_20251226_174046`
  - `results/new_game_theory/shuffle_800_samples/Phi-4-mini-instruct_game_theory_Ultimatum_Game_Responder_20251226_184013`
  - `results/new_game_theory/shuffle_800_samples/Qwen2.5-0.5B-Instruct_game_theory_Escalation_Game_20251226_140237`
  - `results/new_game_theory/shuffle_800_samples/Qwen2.5-0.5B-Instruct_game_theory_Prisoners_Dilemma_20251226_033134`
  - `results/new_game_theory/shuffle_800_samples/Qwen2.5-0.5B-Instruct_game_theory_Stag_Hunt_20251226_125620`
  - `results/new_game_theory/shuffle_800_samples/Qwen2.5-0.5B-Instruct_game_theory_Trust_Game_Trustee_20251226_161007`
  - `results/new_game_theory/shuffle_800_samples/Qwen2.5-0.5B-Instruct_game_theory_Trust_Game_Trustor_20251226_150846`
  - `results/new_game_theory/shuffle_800_samples/Qwen2.5-0.5B-Instruct_game_theory_Ultimatum_Game_Proposer_20251226_171143`
  - `results/new_game_theory/shuffle_800_samples/Qwen2.5-0.5B-Instruct_game_theory_Ultimatum_Game_Responder_20251226_181348`
  - `results/new_game_theory/shuffle_800_samples/Qwen2.5-1.5B-Instruct_game_theory_Escalation_Game_20251226_140847`
  - `results/new_game_theory/shuffle_800_samples/Qwen2.5-1.5B-Instruct_game_theory_Prisoners_Dilemma_20251226_034924`
  - `results/new_game_theory/shuffle_800_samples/Qwen2.5-1.5B-Instruct_game_theory_Stag_Hunt_20251226_130226`
  - `results/new_game_theory/shuffle_800_samples/Qwen2.5-1.5B-Instruct_game_theory_Trust_Game_Trustee_20251226_161555`
  - `results/new_game_theory/shuffle_800_samples/Qwen2.5-1.5B-Instruct_game_theory_Trust_Game_Trustor_20251226_151425`
  - `results/new_game_theory/shuffle_800_samples/Qwen2.5-1.5B-Instruct_game_theory_Ultimatum_Game_Proposer_20251226_171727`
  - `results/new_game_theory/shuffle_800_samples/Qwen2.5-1.5B-Instruct_game_theory_Ultimatum_Game_Responder_20251226_181930`
  - `results/new_game_theory/shuffle_800_samples/Qwen2.5-3B-Instruct_game_theory_Escalation_Game_20251226_141228`
  - `results/new_game_theory/shuffle_800_samples/Qwen2.5-3B-Instruct_game_theory_Prisoners_Dilemma_20251226_035226`
  - `results/new_game_theory/shuffle_800_samples/Qwen2.5-3B-Instruct_game_theory_Stag_Hunt_20251227_065207`
  - `results/new_game_theory/shuffle_800_samples/Qwen2.5-3B-Instruct_game_theory_Trust_Game_Trustee_20251226_161908`
  - `results/new_game_theory/shuffle_800_samples/Qwen2.5-3B-Instruct_game_theory_Trust_Game_Trustor_20251226_151749`
  - `results/new_game_theory/shuffle_800_samples/Qwen2.5-3B-Instruct_game_theory_Ultimatum_Game_Proposer_20251226_172056`
  - `results/new_game_theory/shuffle_800_samples/Qwen2.5-3B-Instruct_game_theory_Ultimatum_Game_Responder_20251226_182202`
  - `results/new_game_theory/shuffle_800_samples/Qwen3-0.6B_game_theory_Escalation_Game_20251226_135445`
  - `results/new_game_theory/shuffle_800_samples/Qwen3-0.6B_game_theory_Prisoners_Dilemma_20251226_032439`
  - `results/new_game_theory/shuffle_800_samples/Qwen3-0.6B_game_theory_Stag_Hunt_20251226_124749`
  - `results/new_game_theory/shuffle_800_samples/Qwen3-0.6B_game_theory_Trust_Game_Trustee_20251226_160102`
  - `results/new_game_theory/shuffle_800_samples/Qwen3-0.6B_game_theory_Trust_Game_Trustor_20251226_145959`
  - `results/new_game_theory/shuffle_800_samples/Qwen3-0.6B_game_theory_Ultimatum_Game_Proposer_20251226_170218`
  - `results/new_game_theory/shuffle_800_samples/Qwen3-0.6B_game_theory_Ultimatum_Game_Responder_20251226_180622`
  - `results/new_game_theory/shuffle_800_samples/Qwen3-1.7B_game_theory_Escalation_Game_20251226_135655`
  - `results/new_game_theory/shuffle_800_samples/Qwen3-1.7B_game_theory_Prisoners_Dilemma_20251226_032712`
  - `results/new_game_theory/shuffle_800_samples/Qwen3-1.7B_game_theory_Stag_Hunt_20251226_125024`
  - `results/new_game_theory/shuffle_800_samples/Qwen3-1.7B_game_theory_Trust_Game_Trustee_20251226_160340`
  - `results/new_game_theory/shuffle_800_samples/Qwen3-1.7B_game_theory_Trust_Game_Trustor_20251226_150235`
  - `results/new_game_theory/shuffle_800_samples/Qwen3-1.7B_game_theory_Ultimatum_Game_Proposer_20251226_170511`
  - `results/new_game_theory/shuffle_800_samples/Qwen3-1.7B_game_theory_Ultimatum_Game_Responder_20251226_180833`
  - `results/new_game_theory/shuffle_800_samples/Qwen3-4B_game_theory_Escalation_Game_20251226_135922`
  - `results/new_game_theory/shuffle_800_samples/Qwen3-4B_game_theory_Prisoners_Dilemma_20251226_034600`
  - `results/new_game_theory/shuffle_800_samples/Qwen3-4B_game_theory_Stag_Hunt_20251227_064352`
  - `results/new_game_theory/shuffle_800_samples/Qwen3-4B_game_theory_Trust_Game_Trustee_20251226_160638`
  - `results/new_game_theory/shuffle_800_samples/Qwen3-4B_game_theory_Trust_Game_Trustor_20251226_150516`
  - `results/new_game_theory/shuffle_800_samples/Qwen3-4B_game_theory_Ultimatum_Game_Proposer_20251226_170813`
  - `results/new_game_theory/shuffle_800_samples/Qwen3-4B_game_theory_Ultimatum_Game_Responder_20251226_181058`
  - `results/new_game_theory/shuffle_800_samples/gemma-3-1b-it_game_theory_Escalation_Game_20251226_142055`
  - `results/new_game_theory/shuffle_800_samples/gemma-3-1b-it_game_theory_Prisoners_Dilemma_20251226_035739`
  - `results/new_game_theory/shuffle_800_samples/gemma-3-1b-it_game_theory_Stag_Hunt_20251227_070159`
  - `results/new_game_theory/shuffle_800_samples/gemma-3-1b-it_game_theory_Trust_Game_Trustee_20251226_162448`
  - `results/new_game_theory/shuffle_800_samples/gemma-3-1b-it_game_theory_Trust_Game_Trustor_20251226_152326`
  - `results/new_game_theory/shuffle_800_samples/gemma-3-1b-it_game_theory_Ultimatum_Game_Proposer_20251226_172709`
  - `results/new_game_theory/shuffle_800_samples/gemma-3-1b-it_game_theory_Ultimatum_Game_Responder_20251226_182716`
  - `results/new_game_theory/shuffle_800_samples/gemma-3-270m-it_game_theory_Escalation_Game_20251226_141553`
  - `results/new_game_theory/shuffle_800_samples/gemma-3-270m-it_game_theory_Prisoners_Dilemma_20251226_035543`
  - `results/new_game_theory/shuffle_800_samples/gemma-3-270m-it_game_theory_Stag_Hunt_20251226_130821`
  - `results/new_game_theory/shuffle_800_samples/gemma-3-270m-it_game_theory_Trust_Game_Trustee_20251226_162247`
  - `results/new_game_theory/shuffle_800_samples/gemma-3-270m-it_game_theory_Trust_Game_Trustor_20251226_152135`
  - `results/new_game_theory/shuffle_800_samples/gemma-3-270m-it_game_theory_Ultimatum_Game_Proposer_20251226_172423`
  - `results/new_game_theory/shuffle_800_samples/gemma-3-270m-it_game_theory_Ultimatum_Game_Responder_20251226_182459`
  - `results/new_game_theory/shuffle_800_samples/gemma-3-4b-it_game_theory_Escalation_Game_20251227_073702`
  - `results/new_game_theory/shuffle_800_samples/gemma-3-4b-it_game_theory_Prisoners_Dilemma_20251226_041123`
  - `results/new_game_theory/shuffle_800_samples/gemma-3-4b-it_game_theory_Stag_Hunt_20251227_071641`
  - `results/new_game_theory/shuffle_800_samples/gemma-3-4b-it_game_theory_Trust_Game_Trustee_20251227_081716`
  - `results/new_game_theory/shuffle_800_samples/gemma-3-4b-it_game_theory_Trust_Game_Trustor_20251227_075709`
  - `results/new_game_theory/shuffle_800_samples/gemma-3-4b-it_game_theory_Ultimatum_Game_Proposer_20251226_173121`
  - `results/new_game_theory/shuffle_800_samples/gemma-3-4b-it_game_theory_Ultimatum_Game_Responder_20251226_183114`

## Method
- For each `(model, game_setting)`, select the latest timestamped run directory.
- Collapse intensity by averaging `ratio` over all intensities present.
- Compute `delta_vs_neutral = ratio(emotion) - ratio(neutral)` for each option/behavior.
- Summarize best/worst emotion deltas and `delta_range = best - worst`.
- In per-game tables, show deltas for all emotions (vs neutral), ranked by Δ descending.
- When `raw_results.json` is available, annotate each emotion as `emo:Δ{stars}[ci_low,ci_high]`.
  - `{stars}` uses Benjamini–Hochberg FDR per game-setting (within Option table / Behavior table).

## Outputs
- Option CSV: `result_analysis/_tmp_reports/new_game_theory_shuffle_800_samples/option_impacted_by_emo_vs_neutral_latest.csv`
- Behavior CSV: `result_analysis/_tmp_reports/new_game_theory_shuffle_800_samples/behavior_impacted_emo_vs_neutral_latest.csv`

## Strongest Option Effects (Top 3 by delta_range)
| game_setting | model | option_id | neutral | best (Δ) | worst (Δ) | range |
|---|---|---|---:|---|---|---:|
| Trust_Game_Trustee | Qwen2.5-0.5B-Instruct | 1 | 0.492 | sadness (+0.287) | surprise (-0.148) | 0.435 |
| Trust_Game_Trustee | Qwen2.5-0.5B-Instruct | 3 | 0.271 | surprise (+0.235) | sadness (-0.182) | 0.417 |
| Escalation_Game | Qwen2.5-0.5B-Instruct | 1 | 0.601 | happiness (+0.284) | disgust (-0.108) | 0.392 |

## Per Game Setting Option (Top 1 by delta_range)
### Escalation_Game
| model | option_id | neutral | all emotion deltas (Δ vs neutral) | range |
|---|---|---:|---|---:|
| Qwen2.5-0.5B-Instruct | 1 | 0.601 | happiness:+0.162[-0.081,+0.378]; fear:+0.080[-0.280,+0.400]; surprise:+0.000[-0.158,+0.211]; disgust:-0.167[-0.500,+0.167] | 0.392 |

### Prisoners_Dilemma
| model | option_id | neutral | all emotion deltas (Δ vs neutral) | range |
|---|---|---:|---|---:|
| Phi-4-mini-instruct | 2 | 0.429 | surprise:+0.000[-0.130,+0.120]; fear:-0.020[-0.150,+0.110]; sadness:-0.174[-0.370,+0.000]; happiness:-0.214[-0.357,-0.071]; disgust:-0.500[-1.000,+0.000] | 0.292 |

### Stag_Hunt
| model | option_id | neutral | all emotion deltas (Δ vs neutral) | range |
|---|---|---:|---|---:|
| Qwen2.5-0.5B-Instruct | 1 | 0.612 | happiness:+0.222*[+0.069,+0.347]; disgust:+0.091[-0.136,+0.318]; surprise:+0.067[-0.133,+0.267]; fear:-0.083[-0.306,+0.083] | 0.388 |

### Trust_Game_Trustee
| model | option_id | neutral | all emotion deltas (Δ vs neutral) | range |
|---|---|---:|---|---:|
| Qwen2.5-0.5B-Instruct | 1 | 0.492 | fear:+0.167[-0.333,+0.667]; happiness:-0.123*[-0.217,-0.044] | 0.435 |

### Trust_Game_Trustor
| model | option_id | neutral | all emotion deltas (Δ vs neutral) | range |
|---|---|---:|---|---:|
| Llama-3.2-1B-Instruct | 1 | 0.665 | anger:-0.114117;happiness:-0.425215;fear:-0.435732;sadness:-0.460077;disgust:-0.478392;surprise:-0.483367 | 0.369 |

### Ultimatum_Game_Proposer
| model | option_id | neutral | all emotion deltas (Δ vs neutral) | range |
|---|---|---:|---|---:|
| Llama-3.2-3B-Instruct | 1 | 0.485 |  | 0.370 |

### Ultimatum_Game_Responder
| model | option_id | neutral | all emotion deltas (Δ vs neutral) | range |
|---|---|---:|---|---:|
| Llama-3.2-1B-Instruct | 1 | 0.623 | surprise:+0.171023;fear:+0.039359;anger:-0.001276;sadness:-0.011476;happiness:-0.021602;disgust:-0.146361 | 0.317 |

## Strongest Behavior Effects (Top 3 by delta_range)
| game_setting | model | behavior_label | neutral | best (Δ) | worst (Δ) | range |
|---|---|---|---:|---|---|---:|
| Ultimatum_Game_Responder | Phi-4-mini-instruct | accept | 0.709 | happiness (+0.278) | anger (-0.627) | 0.905 |
| Ultimatum_Game_Responder | Phi-4-mini-instruct | reject | 0.291 | anger (+0.627) | happiness (-0.278) | 0.905 |
| Ultimatum_Game_Responder | Llama-3.2-3B-Instruct | accept | 0.861 | happiness (+0.138) | disgust (-0.765) | 0.902 |

## Per Game Setting Behavior (Top 1 by delta_range)
### Escalation_Game
| model | behavior_label | neutral | all emotion deltas (Δ vs neutral) | range |
|---|---|---:|---|---:|
| Llama-3.2-3B-Instruct | escalation | 0.349 | fear:+0.000[+0.000,+0.000] | 0.686 |

### Prisoners_Dilemma
| model | behavior_label | neutral | all emotion deltas (Δ vs neutral) | range |
|---|---|---:|---|---:|
| Phi-4-mini-instruct | cooperate | 0.899 | fear:+0.020[-0.020,+0.060]; surprise:+0.020[-0.010,+0.060]; happiness:+0.014[+0.000,+0.043]; disgust:+0.000[+0.000,+0.000]; sadness:+0.000[+0.000,+0.000] | 0.336 |

### Stag_Hunt
| model | behavior_label | neutral | all emotion deltas (Δ vs neutral) | range |
|---|---|---:|---|---:|
| Llama-3.2-3B-Instruct | cooperate | 0.957 |  | 0.518 |

### Trust_Game_Trustee
| model | behavior_label | neutral | all emotion deltas (Δ vs neutral) | range |
|---|---|---:|---|---:|
| Llama-3.2-3B-Instruct | return_none | 0.056 |  | 0.868 |

### Trust_Game_Trustor
| model | behavior_label | neutral | all emotion deltas (Δ vs neutral) | range |
|---|---|---:|---|---:|
| Llama-3.2-3B-Instruct | trust_none | 0.062 |  | 0.804 |

### Ultimatum_Game_Proposer
| model | behavior_label | neutral | all emotion deltas (Δ vs neutral) | range |
|---|---|---:|---|---:|
| Phi-4-mini-instruct | offer_high | 0.774 | sadness:+0.500[+0.167,+0.833]; surprise:+0.000[+0.000,+0.000]; happiness:-0.333[-1.000,+0.000] | 0.472 |

### Ultimatum_Game_Responder
| model | behavior_label | neutral | all emotion deltas (Δ vs neutral) | range |
|---|---|---:|---|---:|
| Phi-4-mini-instruct | accept | 0.709 | happiness:+0.291***[+0.221,+0.366]; surprise:+0.221***[+0.171,+0.279]; sadness:+0.206***[+0.129,+0.284]; fear:+0.124***[+0.053,+0.187]; disgust:-0.917***[-1.000,-0.750] | 0.905 |

