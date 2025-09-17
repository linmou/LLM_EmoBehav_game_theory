import json 

with open('data/emotions_human_AIstyle.json') as f:
    emo2stimu = json.load(f)
    
for emo, stimulus in emo2stimu.items():
    with open(f'data/stimulus/crowd-enVent-transformed/{emo}.json', 'w') as f:
        json.dump(stimulus, f, indent=8)
        