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

listKeepedFeatures = []

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

    pearson_scores = {i: abs(pearsonr(arrSelectTrainData[:, i], arrRawTrainLabel)[0]) for i in range(arrSelectTrainData.shape[1])}
    sortedIndices = sorted(pearson_scores, key=pearson_scores.get, reverse=True)

    if strModelPre.endswith("RF"):
        model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=10)
    elif strModelPre.endswith("GBRT"):
        model = GradientBoostingRegressor(n_estimators=100, max_depth=8, random_state=10, learning_rate=0.2, alpha=0.9,
                                          criterion='squared_error', n_iter_no_change=10)
    elif strModelPre.endswith("XGBoost"):
        model = XGBRegressor(n_estimators=100, max_depth=8, min_child_weight=5.0, gamma=5.0, random_state=10)
    elif strModelPre.endswith("GPR"):
        model = GaussianProcessRegressor(kernel=1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-10, 1e10), nu=1.5), alpha=1.0, random_state=10)
    elif "KRR" in strModelPre:
        model = KernelRidge(kernel=Matern(length_scale=1.0), gamma=1.0)
    elif "LGBM" in strModelPre:
        model = LGBMRegressor(boosting_type='dart', num_leaves=20, max_depth=8, learning_rate=0.1, n_estimators=100,
                              objective='regression', min_child_weight=1.0, random_state=10)
    elif "ExRF" in strModelPre:
        model = ExtraTreeRegressor(criterion='squared_error', splitter='best', max_depth=8, random_state=10)

    arrColDatas = arrSelectTrainData[:, sortedIndices[0]:sortedIndices[0]+1]
    bestScore = cross_val_score(model, arrColDatas, arrRawTrainLabel, scoring=curSelectScoring, cv=10, n_jobs=1).mean()

    listDel = []
    for i in sortedIndices[1:]:
        nextCol = arrSelectTrainData[:, i:i+1]
        combined = np.hstack((arrColDatas, nextCol))
        newScore = cross_val_score(model, combined, arrRawTrainLabel, scoring=curSelectScoring, cv=10, n_jobs=1).mean()
        if newScore - bestScore < 0.02:
            listDel.append(i)
        else:
            bestScore = newScore
            arrColDatas = combined

    for i in sorted(listDel, reverse=True):
        arrSelectTrainData = np.delete(arrSelectTrainData, i, axis=1)
        arrSelectFeatures = np.delete(arrSelectFeatures, i, axis=0)

    return arrSelectTrainData, arrSelectFeatures
