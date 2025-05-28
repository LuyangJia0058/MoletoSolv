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



