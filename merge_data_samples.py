import json
import os
from pathlib import Path

from games.game_configs import GAME_CONFIGS


for game_name in GAME_CONFIGS.keys():
    data_ls = []
    for file in Path(f"groupchat/scenarios/{game_name}").glob("*.json"):
        with open(file, "r") as f:
            data = json.load(f)
            data['game_name'] = game_name
            data_ls.append(data)

    with open(f"groupchat/scenarios/{game_name}_all_data_samples.json", "w") as f:
        json.dump(data_ls, f, indent=4)