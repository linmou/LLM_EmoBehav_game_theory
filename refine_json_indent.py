import json
import sys

target_json = sys.argv[1]

with open(target_json, "r") as f:
    data = json.load(f)


with open(target_json, "w") as f:
    json.dump(data, f, indent=4)
print(json.dumps(data, indent=4))
