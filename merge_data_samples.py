import json
import os
from pathlib import Path
        
data_ls = []
for file in Path("groupchat/scenarios/Prisoners_Dilemma").glob("*.json"):
    with open(file, "r") as f:
        data = json.load(f)
        data_ls.append(data)

with open("groupchat/scenarios/Prisoners_Dilemma/data_samples.json", "w") as f:
    json.dump(data_ls, f, indent=4)