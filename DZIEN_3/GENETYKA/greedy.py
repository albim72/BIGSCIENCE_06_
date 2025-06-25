#przykład 1  -> heurystyka - Greedy Algorithm
from unittest import result

items = [
    {"name":"złoto","value":100,"weight":10},
    {"name":"srebro","value":30,"weight":6},
    {"name":"diamenty","value":155,"weight":12},
]

capacity = 15

def greedy(items, capacity):
    items = sorted(items, key=lambda x: x["value"]/x["weight"], reverse=True)
    result = []
    total_value = 0
    for item in items:
        if item["weight"] <= capacity:
            result.append(item)
            total_value += item["value"]
    return total_value,result

tot_value = greedy(items, capacity)[0]
res = greedy(items, capacity)[1]


print(greedy(items, capacity))
print(f"wybrane przedmioty:{res}")
print(f"łączna wartość wybranych przedmiotu:{tot_value}")

