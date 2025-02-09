import json
import os
from pathlib import Path

from games.game_configs import GAME_CONFIGS


for game_name in GAME_CONFIGS.keys():
    game_name_str = game_name.value
    data_ls = []
    if Path(f"groupchat/scenarios/{game_name_str}_all_data_samples.json").exists():
        continue
    for file in Path(f"groupchat/scenarios/{game_name_str}").glob("*.json"):
        with open(file, "r") as f:
            data = json.load(f)
            data['game_name'] = game_name_str
            data_ls.append(data)

    with open(f"groupchat/scenarios/{game_name_str}_all_data_samples.json", "w") as f:
        json.dump(data_ls, f, indent=4)
    print(f"Merged {game_name_str} data samples")