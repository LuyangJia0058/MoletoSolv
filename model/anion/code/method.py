import math
import numpy as np

def compareTwoListDiff(listData1, listData2):
    listNotIn2 = []
    for item in listData1:
        if item not in listData2:
            listNotIn2.append(item)

    listNotIn1 = []
    for item in listData2:
        if item not in listData1:
            listNotIn1.append(item)

    return listNotIn1,listNotIn2

def sortDictions(dictTarget, strType):

    if strType == "keys":
        listKeys = list(dictTarget.keys())
        listKeys.sort()
        return listKeys

    listValues = []
    for value in dictTarget.values():
        listValues.append(value)

    listValues.sort()
    listKeys = []
    for value in listValues:
        for key in dictTarget.keys():
            if math.isclose(value, dictTarget[key], rel_tol=1e-6) and key not in listKeys:
                listKeys.append(key)
                break

    return listKeys

