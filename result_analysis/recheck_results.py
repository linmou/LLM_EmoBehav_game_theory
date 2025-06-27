import json

target_file = "results/RepEng/Prisoners_Dilemma_New_Template_vllm_Llama-3.1-8B-Instruct_20250425_022053/exp_results.json"

with open(target_file, 'r') as f:
    data = json.load(f)

for item in data:

    if item['category'] == 1 and item['scenario'] == "Concert Ticket Turmoil: The Seller's Strategy":
        print(item['emotion'])
        print(item['output'])
    #     print(item['emotion'])
    #     print(item['output'])
    #     print('='*100)