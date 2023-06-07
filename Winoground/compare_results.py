import json
import os

dict1_path = "output/209wino/winoground/winoground_detailed_0"
dict2_path = "output/baseline2/winoground/winoground_detailed_0"

f = open(dict1_path)
dict1 = json.load(f)
f = open(dict2_path)
dict2 = json.load(f)
positives = []
negatives = []
for k in dict1:
    result1 = dict1[k]["group"]
    result2 = dict2[k]["group"]
    if result1 == True and result2 == False:
        positives.append(k)
    if result2 == True and result1 == False:
        negatives.append(k)

print(positives)
print(negatives)
    