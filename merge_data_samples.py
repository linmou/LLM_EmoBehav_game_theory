import json
import os
from pathlib import Path

from constants import GameNames
from games.game_configs import GAME_CONFIGS

output_file_format = "data_creation/scenario_creation/langgraph_creation/{game_name_str}_all_data_samples.json"


def merge_data_samples(data_folder, game_name_str):
    data_ls = []
    for file in Path(data_folder).glob("*.json"):
        with open(file, "r") as f:
            data = json.load(f)
            data["game_name"] = game_name_str
            data_ls.append(data)

    with open(output_file_format.format(game_name_str=game_name_str), "w") as f:
        json.dump(data_ls, f, indent=4)
    print(f"Merged {game_name_str} data samples")

    return data_ls


game_name = GameNames.PRISONERS_DILEMMA
game_name_str = game_name.value
if not Path(output_file_format.format(game_name_str=game_name_str)).exists():
    data_folder = "data_creation/scenario_creation/langgraph_creation/scenarios/Prisoners_Dilemma_20250411"
    merge_data_samples(data_folder, game_name_str)
