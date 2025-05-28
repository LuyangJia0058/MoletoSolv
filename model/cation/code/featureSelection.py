import numpy as np
import math
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.inspection import permutation_importance
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import method

listKeepedFeatures = ['strGasDeltaG']

def selFeaturesByVariance(arrRawTrainData, arrRawFeatures):
    arrSelectTrainData = arrRawTrainData
    arrSelectFeatures = arrRawFeatures
    listVariance = [np.var(arrSelectTrainData[:, i]) for i in range(arrSelectTrainData.shape[1])]

    for i in reversed(range(len(listVariance))):
        if listVariance[i] <= 0.01:
            arrSelectTrainData = np.delete(arrSelectTrainData, i, axis=1)
            arrSelectFeatures = np.delete(arrSelectFeatures, i, axis=0)

    return arrSelectTrainData, arrSelectFeatures

def selFeaturesByPearsonMap(arrRawTrainData, arrRawTrainLabel, arrRawFeatures):
    arrSelectTrainData = arrRawTrainData
    arrSelectFeatures = arrRawFeatures
    nCols = arrSelectTrainData.shape[1]
    arrPearsonsMap = np.empty((nCols, nCols), dtype=float)

    for i in range(nCols):
        for j in range(nCols):
            r, _ = pearsonr(arrSelectTrainData[:, i], arrSelectTrainData[:, j])
            arrPearsonsMap[i][j] = abs(r)

    listDelCols = []
    for i in range(nCols):
        if i in listDelCols:
            continue
        for j in range(nCols):
            if j == i:
                continue
            if arrPearsonsMap[i][j] > 0.9:
                r1 = abs(pearsonr(arrSelectTrainData[:, i], arrRawTrainLabel)[0])
                r2 = abs(pearsonr(arrSelectTrainData[:, j], arrRawTrainLabel)[0])
                if r1 > r2 and j not in listDelCols:
                    listDelCols.append(j)
                elif r1 <= r2 and i not in listDelCols:
                    listDelCols.append(i)

    for i in sorted(set(listDelCols), reverse=True):
        if arrSelectFeatures[i] in listKeepedFeatures:
            continue
        arrSelectTrainData = np.delete(arrSelectTrainData, i, axis=1)
        arrSelectFeatures = np.delete(arrSelectFeatures, i, axis=0)

    return arrSelectTrainData, arrSelectFeatures

def selFeaturesByPearsonWithLabel(arrRawTrainData, arrRawTrainLabel, arrRawFeatures):
    arrSelectTrainData = arrRawTrainData
    arrSelectFeatures = arrRawFeatures
    dictPearsonCoeff = {}

    for i in range(arrSelectTrainData.shape[1]):
        if arrSelectFeatures[i] in listKeepedFeatures:
            dictPearsonCoeff[i] = 0.0
        else:
            r, _ = pearsonr(arrSelectTrainData[:, i], arrRawTrainLabel)
            dictPearsonCoeff[i] = abs(r)

    listSortedKeys = method.sortDictions(dictPearsonCoeff, "Value")
    listSortedKeys.reverse()
    listDelIndex = sorted(listSortedKeys[40:], reverse=True)

    for i in listDelIndex:
        if arrSelectFeatures[i] in listKeepedFeatures:
            continue
        arrSelectTrainData = np.delete(arrSelectTrainData, i, axis=1)
        arrSelectFeatures = np.delete(arrSelectFeatures, i, axis=0)

    return arrSelectTrainData, arrSelectFeatures

def selFeaturesForwardViaModel(arrRawTrainData, arrRawTrainLabel, arrRawFeatures, strModelPre, curSelectScoring):

    arrSelectTrainData = arrRawTrainData
    arrSelectFeatures = arrRawFeatures

    nCols = arrSelectTrainData.shape[1]
    listPearsonCoeff = []
    listPearsonP = []
    for nIndex in range(0,nCols,1):
        arrColDatas = arrSelectTrainData[:,nIndex]
        r,pValue = pearsonr(arrColDatas,arrRawTrainLabel)
        listPearsonCoeff.append(r)
        listPearsonP.append(pValue)

    dictValidCoeff = {}
    for nIndex in range(len(listPearsonCoeff)-1,-1, -1):
        dictValidCoeff[nIndex] = math.fabs(listPearsonCoeff[nIndex])

    listSortedKeys = method.sortDictions(dictValidCoeff, "value")
    listSortedKeys.reverse()

    nTreeDepth = 8
    curModel = ""
    if strModelPre.endswith("RF"):
        curModel = RandomForestRegressor(n_estimators=100,max_depth=nTreeDepth,random_state=10)
    elif strModelPre.endswith("GBRT"):
        curModel = GradientBoostingRegressor(n_estimators=100,max_depth=nTreeDepth,random_state=10,learning_rate=0.2, alpha=0.9,
                                             criterion='squared_error', n_iter_no_change=10)
    elif strModelPre.endswith("XGBoost"):
        curModel = XGBRegressor(n_estimators=100,max_depth=nTreeDepth,min_child_weight=5.0,gamma=5.0,random_state=10)
    elif strModelPre.find("KRR") > -1:
        Maternkernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-10, 1e10), nu=1.5)
        curModel = KernelRidge(kernel=Maternkernel, gamma = 1.0)
    elif strModelPre.find("LGBM") > -1:
        curModel = LGBMRegressor(boosting_type='dart', num_leaves=20, max_depth=nTreeDepth,learning_rate=0.1,n_estimators=100,
                                  objective='regression',min_child_weight=1.0,random_state=10)
    elif strModelPre.find("ExRF") > -1:
        curModel = ExtraTreeRegressor(criterion='squared_error', splitter='best', max_depth=nTreeDepth,random_state=10)

    nCols = len(listSortedKeys)
    arrColDatas = arrSelectTrainData[:,listSortedKeys[0]:listSortedKeys[0]+1] #Pearson值最高的特征
    scores = cross_val_score(curModel, arrColDatas, arrRawTrainLabel, scoring=curSelectScoring, cv = 10, n_jobs=1)
    totalVariance = np.var(scores)
    totalScore = np.mean(scores)

    listDelFeatures = []
    for nIndex in range(1,nCols):
        nColIndex = listSortedKeys[nIndex]
        curColDatas = arrSelectTrainData[:,nColIndex:nColIndex+1] 
        arrTmpDatas = np.hstack((arrColDatas,curColDatas))
        scores = cross_val_score(curModel, arrTmpDatas, arrRawTrainLabel, scoring = curSelectScoring, cv = 10, n_jobs=-1)
        meanScore = np.mean(scores)
        curVariance = np.var(scores)
        if meanScore - totalScore < 0.02 and nColIndex not in listDelFeatures and arrSelectFeatures[nColIndex] not in listKeepedFeatures:
            listDelFeatures.append(nColIndex)
        else: 
            totalScore = meanScore
            totalVariance = curVariance
            arrColDatas = arrTmpDatas
 
    listDelFeatures.sort()
    listDelFeatures.reverse()
    for nIndex in listDelFeatures:
        arrSelectFeatures = np.delete(arrSelectFeatures, nIndex, axis=0)
        arrSelectTrainData = np.delete(arrSelectTrainData, nIndex, axis=1)

    return arrSelectTrainData,arrSelectFeatures

