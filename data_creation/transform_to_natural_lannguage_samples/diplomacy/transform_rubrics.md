Transform the structuralized data into natural language described decision-making scenario. 
You should following these rules: 
1. Write in a historical background rather than a game
2. Use the full name of province.
3. Describe the payoffs of each choice, including  benefits and drawbacks for all stake holders.
4. Add 'You are a <participant_name> commander' as the first sentence of description.
5. For multi-turn cases, no need to describe the history in the 'description' field. Just describe them in 'previous_actions'
6. for the previous_actions : 
6.1 - generate canonicalized action descriptions from the commitment level
6.2 - keep them family-specific and template-based
6.3 - do not fabricate exact provinces or exact support origins unless that detail is part of the decision framing
7. Avoid explicit game-mechanism jargon such as “beauty contest,” “bid,” “variant,” or “payoff matrix”
8.  Keep participant naming consistent across `description`, `participants`, and `previous_actions`
9. In multi-turn cases, `previous_actions` must be non-empty. Each previous round should be a dictionary with `round`, `round_summary`, and a non-empty `actions` list. Each `actions` list must include per-participant dictionaries with `participant` and canonical `action` strings copied exactly from the declared behavior choices.
10. Keep behavior choices as close as possible. Do not use explicit semantic different wording like "Do" vs "Do not" , keep different at slight level. and do not reflect any emotional wording. 
11. In multvior choices for multi-turn cases, do not reflect previous turn decisions e.g. 'xxx like before' which is forbidden. 
