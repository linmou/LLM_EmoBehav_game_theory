import yaml
import json
import pandas as pd
from pathlib import Path

from statistical_engine import analyze_emotion_and_intensity_effects


exp_res_dir_1 = Path("results/escalation_game_previous_actions_1_20250210_014112")
exp_res_dir_2 = Path("results/escalation_game_previous_actions_1_20250210_015819")

raw_analyse_file = 'analysis_results.json'

exp_cfg_file = 'exp_cfg.yaml'

with open(exp_res_dir_1 / exp_cfg_file, 'r') as f:
    exp_cfg_1 = yaml.safe_load(f)

with open(exp_res_dir_2 / exp_cfg_file, 'r') as f:
    exp_cfg_2 = yaml.safe_load(f)

repeat_num_1 = exp_cfg_1['experiment'].pop('repeat')
repeat_num_2 = exp_cfg_2['experiment'].pop('repeat')
ttl_repeat_num = repeat_num_1 + repeat_num_2

assert exp_cfg_1 == exp_cfg_2

# create new directory
def max_common_prefix(str1, str2):
    prefix = ""
    for c1, c2 in zip(str1, str2):
        if c1 == c2:
            prefix += c1
        else:
            break
    return prefix


exp_res_dir_3 = Path(f'{max_common_prefix(str(exp_res_dir_1), str(exp_res_dir_2))}_Merged')
print(f'merge into new directory: {exp_res_dir_3}')
exp_res_dir_3.mkdir(parents=True, exist_ok=True)
with open(exp_res_dir_3 / 'merge_info.txt', 'w') as f:
    f.write(f'merge {exp_res_dir_1} and {exp_res_dir_2} into {exp_res_dir_3}')



# begin merge

exp_cfg_1['experiment']['repeat'] = ttl_repeat_num
with open(exp_res_dir_3 / exp_cfg_file, 'w') as f:
    yaml.dump(exp_cfg_1, f, indent=2)
    
total_records = []
for json_file in exp_res_dir_1.glob('*.json'):
    individual_record_json = json_file.name
    
    if individual_record_json == raw_analyse_file:
        continue

    with open(json_file, 'r') as f:
        individual_record = json.load(f)

    with open(exp_res_dir_2 / individual_record_json, 'r') as f:
        individual_record_2 = json.load(f)

    for record in range(len(individual_record_2)):
        individual_record_2[record]['repeat_num'] = individual_record_2[record]['repeat_num'] + repeat_num_1
        
    individual_record.extend(individual_record_2)

    with open(exp_res_dir_3 / individual_record_json, 'w') as f:
        json.dump(individual_record, f, indent=4)

    total_records.extend(individual_record)

df = pd.DataFrame(total_records).replace('None','Neutral')
df.to_csv(exp_res_dir_3 / 'all_output_samples.csv', index=False)
print(f'total records: {len(total_records)}')

# statistical analysis

res = analyze_emotion_and_intensity_effects(str(exp_res_dir_3 / 'all_output_samples.csv'), str(exp_res_dir_3))

with open(exp_res_dir_3 / 'analysis_results.json', 'w') as f:
    json.dump(res, f, indent=4)
    
print(f'analysis results saved to {exp_res_dir_3 / "analysis_results.json"}')
