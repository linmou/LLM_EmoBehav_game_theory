import json

data_path = "data_creation/scenario_creation/langgraph_creation/Prisoners_Dilemma_all_data_samples.json"

with open(data_path, "r") as f:
    data = json.load(f)

first_k_data = data[:1000]

with open("data_creation/scenario_creation/langgraph_creation/Prisoners_Dilemma_all_data_samples_first_1000.json", "w") as f:
    json.dump(first_k_data, f, indent=4)
