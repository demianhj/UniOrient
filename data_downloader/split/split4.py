
import json
import os


data_path= './categories_counts.json'
with open(data_path, 'r') as f:
    data= json.load(f)




bins = {i: {"sum": 0, "items": {}} for i in range(1, 5)}
for name, count in sorted(data.items(), key=lambda x: x[1], reverse=True):
    target = min(bins, key=lambda k: bins[k]["sum"])
    bins[target]["items"][name] = count
    bins[target]["sum"] += count


for i, content in bins.items():
    filename = f"set_{i}.json"
    with open(filename, "w") as f:
        json.dump(content["items"], f, indent=2)
    print(f"Saved Set {i} → {filename}: "
          f"{len(content['items'])} categories, total instances = {content['sum']}")