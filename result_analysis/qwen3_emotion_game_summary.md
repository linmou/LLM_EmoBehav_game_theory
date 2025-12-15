# Qwen3 Emotion Impact by Model Size (none-thinking series)

## Scope

* **Inputs:** non-sanity `stats_analysis.json` files under `results/Qwen3_Series_none_thinking`, excluding `crowd-enVent-transform/` reruns.
* **Models covered:** Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B-AWQ, Qwen3-8B-AWQ, Qwen3-14B-AWQ.
* **Metrics:** choice ratios per emotion; deltas reported in percentage points (pp) versus the run’s neutral baseline.
* **References:** each subsection cites the exact stats file so the raw counts and chi-square outputs remain traceable. For full metric-by-emotion grids per model, see `result_analysis/qwen3_emotion_deltas_full.md`.

---

## Model-Level Findings (with psychological sketch)

### Qwen3-0.6B

* **Neutral baselines:**

  * Prisoner’s Dilemma: defection 20.6%
  * Escalation Game: escalation 32.6%
  * Trust Game (Trustor): high trust 63.3%, no trust 3.9%
  * Ultimatum Game (Proposer): fair split 54.0%, selfish 42.7%
* **Stats files:**

  * `results/Qwen3_Series_none_thinking/Qwen3_Series_none_thinking_Prisoners_Dilemma_Qwen3-0.6B_20250803_014318/stats_analysis.json`
  * `results/Qwen3_Series_none_thinking/Qwen3_Series_none_thinking_Escalation_Game_Qwen3-0.6B_20250803_032557/stats_analysis.json`
  * `results/Qwen3_Series_none_thinking/Qwen3_Series_none_thinking_Trust_Game_Trustor_Qwen3-0.6B_20250803_040116/stats_analysis.json`
  * `results/Qwen3_Series_none_thinking/Qwen3_Series_none_thinking_Ultimatum_Game_Proposer_Qwen3-0.6B_20250803_060020/stats_analysis.json`
* **Emotion Impacts (significant only):**

  * Anger: Trustor → High Trust -46.77 pp, No Trust +46.23 pp
  * Disgust: Escalation → Escalate -8.42 pp; Ultimatum Proposer → Fair Split +6.33 pp, Selfish Split -10.00 pp
  * Fear: Prisoners Dilemma → Defection -5.39 pp; Escalation → Escalate -6.96 pp; Trustor → High Trust +7.00 pp, No Trust -2.04 pp
  * Happiness: Prisoners Dilemma → Defection +9.47 pp; Trustor → High Trust -4.57 pp, No Trust -2.48 pp
  * Sadness: Prisoners Dilemma → Defection -5.31 pp; Escalation → Escalate +10.50 pp; Trustor → High Trust -12.12 pp, No Trust +2.14 pp
  * Surprise: Prisoners Dilemma → Defection +6.77 pp; Escalation → Escalate -7.94 pp; Trustor → High Trust -12.91 pp, No Trust +5.21 pp; Ultimatum Proposer → Fair Split +9.33 pp, Selfish Split -10.67 pp
* **Psychological Interpretation:**
  At this smaller scale, strong negative emotions like anger drastically reduce trust (aligning with **appraisal theory**, where anger cues hostility and blame). Fear shows mixed effects—sometimes reducing risk-taking (lower defection, escalation) but also boosting trust, consistent with **prospect theory’s risk sensitivity under fear**. Happiness increases risky cooperation (higher defection) but undermines trust, supporting the **affect-as-information hypothesis** that positive mood can bias judgment toward heuristic shortcuts. Sadness magnifies escalation and reduces trust, in line with findings that sadness narrows attentional focus.

---

### Qwen3-1.7B

* **Neutral baselines:**

  * Prisoner’s Dilemma: defection 1.86%
  * Escalation Game: escalation 60.4%
  * Trust Game (Trustor): high trust 21.3%, no trust 23.1%
  * Ultimatum Game (Proposer): fair split 48.3%, selfish 42.3%
* **Stats files:**

  * `results/Qwen3_Series_none_thinking/Qwen3_Series_none_thinking_Prisoners_Dilemma_Qwen3-1.7B_20250803_020553/stats_analysis.json`
  * `results/Qwen3_Series_none_thinking/Qwen3_Series_none_thinking_Escalation_Game_Qwen3-1.7B_20250803_032942/stats_analysis.json`
  * `results/Qwen3_Series_none_thinking/Qwen3_Series_none_thinking_Trust_Game_Trustor_Qwen3-1.7B_20250803_040951/stats_analysis.json`
  * `results/Qwen3_Series_none_thinking/Qwen3_Series_none_thinking_Ultimatum_Game_Proposer_Qwen3-1.7B_20250803_060240/stats_analysis.json`
* **Emotion Impacts (significant only):**

  * Anger: Trustor → High Trust -5.06 pp, No Trust +6.70 pp
  * Disgust: Escalation → Escalate -6.72 pp; Trustor → High Trust +4.57 pp, No Trust -3.43 pp
  * Fear: Trustor → High Trust -2.98 pp, No Trust -2.63 pp
  * Happiness: Escalation → Escalate +6.59 pp; Trustor → High Trust +1.89 pp, No Trust -5.36 pp
  * Sadness: Trustor → High Trust -0.89 pp, No Trust -5.71 pp
  * Surprise: Trustor → High Trust -1.94 pp, No Trust -3.43 pp
* **Psychological Interpretation:**
  At 1.7B, anger still undermines trust but with smaller magnitudes, suggesting **emotional regulation capacity** is somewhat better. Disgust reduces escalation, resonating with the **moral disgust framework** where aversion discourages destructive conflict. Happiness promotes escalation, consistent with **broaden-and-build theory**, where positive affect can amplify action tendencies. Sadness consistently lowers trust, matching studies that sadness increases caution and withdrawal.

---

### Qwen3-4B-AWQ

* **Neutral baselines:**

  * Prisoner’s Dilemma: defection 2.58%
  * Escalation Game: escalation 57.9%
  * Trust Game (Trustor): high trust 60.0%, no trust 0.05%
  * Ultimatum Game (Proposer): fair split 46.3%, selfish 39.0%
* **Stats files:**

  * `results/Qwen3_Series_none_thinking/Qwen3_Series_none_thinking_Prisoners_Dilemma_Qwen3-4B-AWQ_20250803_021635/stats_analysis.json`
  * `results/Qwen3_Series_none_thinking/Qwen3_Series_none_thinking_Escalation_Game_Qwen3-4B-AWQ_20250803_033315/stats_analysis.json`
  * `results/Qwen3_Series_none_thinking/Qwen3_Series_none_thinking_Trust_Game_Trustor_Qwen3-4B-AWQ_20250803_042233/stats_analysis.json`
  * `results/Qwen3_Series_none_thinking/Qwen3_Series_none_thinking_Ultimatum_Game_Proposer_Qwen3-4B-AWQ_20250803_060503/stats_analysis.json`
* **Emotion Impacts (significant only):**

  * Anger: Prisoners Dilemma → Defection -1.11 pp; Escalation → Escalate +5.49 pp; Trustor → High Trust +20.31 pp, No Trust +0.35 pp; Ultimatum Proposer → Fair Split -14.00 pp, Selfish Split +15.33 pp
  * Disgust: Prisoners Dilemma → Defection +5.47 pp; Escalation → Escalate -15.51 pp; Trustor → High Trust -32.92 pp, No Trust -0.05 pp; Ultimatum Proposer → Fair Split +18.00 pp, Selfish Split -10.33 pp
  * Fear: Prisoners Dilemma → Defection +4.56 pp; Trustor → High Trust -5.66 pp, No Trust +0.05 pp
  * Happiness: Prisoners Dilemma → Defection -1.98 pp; Escalation → Escalate +9.28 pp; Trustor → High Trust +19.61 pp, No Trust +0.00 pp; Ultimatum Proposer → Fair Split -12.00 pp, Selfish Split +14.33 pp
  * Sadness: Prisoners Dilemma → Defection -1.55 pp; Escalation → Escalate +12.58 pp; Trustor → High Trust +20.95 pp, No Trust +0.20 pp; Ultimatum Proposer → Fair Split -10.00 pp, Selfish Split +12.67 pp
  * Surprise: Prisoners Dilemma → Defection -1.03 pp; Escalation → Escalate -12.45 pp; Trustor → High Trust -8.19 pp, No Trust +1.34 pp; Ultimatum Proposer → Fair Split -9.00 pp, Selfish Split +10.33 pp
* **Psychological Interpretation:**
  By 4B, the model displays more nuanced, sometimes contradictory emotional effects. Anger increases escalation but also boosts high trust—suggesting **dual-process effects** where anger can both fuel assertiveness and signal reliability in social exchanges. Disgust reduces escalation but drastically erodes trust, reflecting its role in **boundary-maintenance psychology**. Happiness amplifies escalation but also strengthens trust, aligning with **broaden-and-build** tendencies. Sadness surprisingly strengthens trust, showing a possible **empathy-inducing mechanism**, where sadness is interpreted as vulnerability, promoting prosocial behavior.

---

### Qwen3-8B-AWQ

* **Neutral baselines:**

  * Prisoner’s Dilemma: defection 4.08%
  * Escalation Game: escalation 80.8%
  * Trust Game (Trustor): high trust 55.6%, no trust 0.20%
  * Ultimatum Game (Proposer): fair split 69.3%, selfish 7.33%
* **Stats files:**

  * `results/Qwen3_Series_none_thinking/Qwen3_Series_none_thinking_Prisoners_Dilemma_Qwen3-8B-AWQ_20250803_023341/stats_analysis.json`
  * `results/Qwen3_Series_none_thinking/Qwen3_Series_none_thinking_Escalation_Game_Qwen3-8B-AWQ_20250803_034010/stats_analysis.json`
  * `results/Qwen3_Series_none_thinking/Qwen3_Series_none_thinking_Trust_Game_Trustor_Qwen3-8B-AWQ_20250803_044502/stats_analysis.json`
  * `results/Qwen3_Series_none_thinking/Qwen3_Series_none_thinking_Ultimatum_Game_Proposer_Qwen3-8B-AWQ_20250803_060731/stats_analysis.json`
* **Emotion Impacts (significant only):**

  * Anger: Prisoners Dilemma → Defection +2.65 pp; Escalation → Escalate -4.52 pp; Trustor → High Trust -3.77 pp, No Trust +0.25 pp; Ultimatum Proposer → Fair Split -10.67 pp, Selfish Split -0.67 pp
  * Disgust: Prisoners Dilemma → Defection +2.18 pp; Escalation → Escalate -16.24 pp; Trustor → High Trust -3.62 pp, No Trust +4.77 pp
  * Fear: Prisoners Dilemma → Defection +4.48 pp; Escalation → Escalate -4.27 pp; Trustor → High Trust +12.61 pp, No Trust +0.94 pp
  * Happiness: Prisoners Dilemma → Defection -1.39 pp; Escalation → Escalate +10.13 pp; Trustor → High Trust -7.89 pp, No Trust -0.20 pp
  * Sadness: Prisoners Dilemma → Defection -2.42 pp; Escalation → Escalate +5.25 pp; Trustor → High Trust +13.26 pp, No Trust -0.20 pp
  * Surprise: Prisoners Dilemma → Defection +1.86 pp; Escalation → Escalate -12.45 pp; Trustor → High Trust -21.70 pp, No Trust +0.89 pp
* **Psychological Interpretation:**
  At 8B, emotional impacts become sharper but polarized. Fear strongly increases trust, fitting with **attachment theory** where fear elicits affiliative responses. Surprise reduces trust substantially, consistent with **uncertainty appraisal models**—surprise disrupts predictability and diminishes willingness to cooperate. Happiness again drives escalation but erodes trust, reinforcing the **heuristic-bias hypothesis** of positive affect.

---

### Qwen3-14B-AWQ

* **Neutral baselines:**

  * Prisoner’s Dilemma: defection 3.65%
  * Escalation Game: escalation 70.0%
  * Trust Game (Trustor): high trust 25.6%, no trust 0.5%
  * Ultimatum Game (Proposer): fair split 72.0%, selfish 5.67%
* **Stats files:**

  * `results/Qwen3_Series_none_thinking/Qwen3_Series_none_thinking_Prisoners_Dilemma_Qwen3-14B-AWQ_20250803_025524/stats_analysis.json`
  * `results/Qwen3_Series_none_thinking/Qwen3_Series_none_thinking_Escalation_Game_Qwen3-14B-AWQ_20250803_034900/stats_analysis.json`
  * `results/Qwen3_Series_none_thinking/Qwen3_Series_none_thinking_Trust_Game_Trustor_Qwen3-14B-AWQ_20250803_051508/stats_analysis.json`
  * `results/Qwen3_Series_none_thinking/Qwen3_Series_none_thinking_Ultimatum_Game_Proposer_Qwen3-14B-AWQ_20250803_061025/stats_analysis.json`
* **Emotion Impacts (significant only):**

  * Anger: Trustor → High Trust -7.00 pp, No Trust +0.35 pp
  * Disgust: Escalation → Escalate -6.35 pp; Trustor → High Trust -4.77 pp, No Trust +0.15 pp
  * Happiness: Escalation → Escalate +7.57 pp; Trustor → High Trust +7.45 pp, No Trust -0.50 pp
  * Sadness: Prisoners Dilemma → Defection +1.55 pp; Escalation → Escalate +5.25 pp; Trustor → High Trust +7.85 pp, No Trust -0.45 pp
* **Psychological Interpretation:**
  At 14B, effects converge toward more human-like emotion patterns. Anger and disgust both reduce trust, supporting **social appraisal theories** where these emotions mark others as threats or contaminants. Happiness elevates both escalation and trust, showing **ambivalence in positive affect**—it motivates bold action but can also sustain cooperation. Sadness consistently boosts trust, which aligns with **empathy-based cooperation theories** where sadness invites supportive responses.
