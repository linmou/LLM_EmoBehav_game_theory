import json

import pandas as pd

with open(
    "data_creation/scenario_creation/langgraph_creation/Prisoners_Dilemma_all_data_samples.json",
    "r",
) as f:
    data = json.load(f)

df = pd.DataFrame(data)
df.drop(columns=["payoff_description"], inplace=True)
df.to_csv(
    "data_creation/scenario_creation/langgraph_creation/Prisoners_Dilemma_all_data_samples.csv",
    index=False,
)
