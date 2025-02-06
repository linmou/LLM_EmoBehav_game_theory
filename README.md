Experiment on llm's emotional reactions for game scenarios
Hypothesis: with different emotion setting in LLM's system prompt, we can observe significant diff in their reactions,
in some game theory scenarios. 

Emotions are in constants.py

follow the three steps
# Data Creation
create data for a specific game scenario
python -m data_creation.create_scenario

# API-based Test
python api_test_engine.py

# Statistic Analysis
python statistical_engine.py