Experiment on llm's emotional reactions for game scenarios
Hypothesis: with different emotion setting in LLM's system prompt, we can observe significant diff in their reactions,
in some game theory scenarios. 

Emotions are in constants.py

follow the three steps
# Data Creation
create data for a specific game scenario

## AG2 based Scenario Creator
python -m data_creation.create_scenario

## LangGraph-based Scenario Creator
a LangGraph-based scenario creation system that offers iterative refinement:



See [Scenario Creation Graph Documentation](doc/scenario_creation_graph.md) for details on how the LangGraph implementation works.

# API-based Test
python api_test_engine.py

python prompt_experiment.py # run the pipeline

# Statistic Analysis
python statistical_engine.py