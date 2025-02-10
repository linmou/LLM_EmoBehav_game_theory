# Research Questions
RQ1. Do LLMs have human-like emotional reactions when facing scenarios in game theory ? (diff between emotions)
RQ2. Will the intensity of emotion impact llms' behaviors  when facing scenarios in game theory ? ( diff between  activation levels of emotion)
RQ3. Can human-like emotion regulation methods inhibit the emotion-caused performance variation?

## EXP insights 
tips to make diff more sig
based on gpt4o:
1. replace 'Alice' by 'you' in the scenario data,
2. add percentage (90%) in system msg to strengthen emotion
3. 'hold your emotion and make the decision' makes NO diff


## Escalating Game

### Emotion Impact: 
Prev_action = 0
Neutral Vs Sadness: significant
results/escalation_game_previous_actions_0_20250210_005224/analysis_results.json

Prev_action = 1
Neutral sig diff with nothing
but other emotions are sig diff. 
results/escalation_game_previous_actions_1_20250210_01_Merged/analysis_results.json

Prev_action = 2
Neutral Vs Anger: significant

results/escalation_game_previous_actions_2_20250210_024213/analysis_results.json

