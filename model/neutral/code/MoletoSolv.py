import os
import sys
import time
import math
import numpy as np
import pandas as pd
import joblib
import csv

from collections import Counter
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, maxabs_scale
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.inspection import permutation_importance

import method
import featureSelection

BASISDIR = "C:/Users/yourname/your_project_path/"

nMaxDepth = 10
dEarlyStop = 0.00
dTol = 3.0 # SVM tolerance

def getIntitialDataSet(strPath):

    print("\nLoading and preprocessing initial dataset...")

    strFilePath = BASISDIR + "features.csv"
    arrRawData = np.genfromtxt(strFilePath, np.dtype(str), delimiter=",")

    nCol = arrRawData.shape[1]-1 
    arrRawFeatures = arrRawData[0,0:nCol]
    arrRawData = np.delete(arrRawData,0, axis=0)
    
    np.random.seed(11)
    np.random.shuffle(arrRawData)

    np.random.seed(9)
    np.random.shuffle(arrRawData)

    np.random.seed(10)
    np.random.shuffle(arrRawData)

    arrRawLabel = arrRawData[:,nCol]
    arrRawData = np.delete(arrRawData, nCol, axis=1)

    arrRawData = arrRawData.astype(np.float64)
    arrRawLabel = arrRawLabel.astype(np.float64)

    strInitialDataFile = strPath + "InitialData.npy"
    strInitialLabelFile = strPath + "InitialLabel.npy"
    strInitialFeatureFile = strPath + "InitialFeature.npy"
    np.save(strInitialDataFile,arrRawData)
    np.save(strInitialLabelFile,arrRawLabel)
    np.save(strInitialFeatureFile,arrRawFeatures)

    return  arrRawData,arrRawLabel,arrRawFeatures

def splitDataSet(arrRawData, arrRawLabel, arrRawFeatures, strPath):
    print("\nSplitting dataset into training and test sets...")

    nSplitType = 1  # 1: random; 2: by solute; 3: by solvent
    arrRawTrainData = []
    arrRawTestData = []
    arrRawTrainLabel = []
    arrRawTestLabel = []

    nRandomSeed = 37

    if nSplitType == 1:
        print("\nSplitting method: random")
        arrRawTrainData, arrRawTestData, arrRawTrainLabel, arrRawTestLabel = train_test_split(
            arrRawData, arrRawLabel, test_size=0.1, random_state=nRandomSeed)

    elif nSplitType == 2:
        print("\nSplitting method: by solute")
        arrRawData_df = pd.DataFrame(arrRawData, columns=arrRawFeatures)
        ClassificationFeatures = "strSASA"
        arrRawDataUniqueClusters = arrRawData_df[ClassificationFeatures].unique()
        arrRawTrainDataUniqueClusters, _ = train_test_split(
            arrRawDataUniqueClusters, test_size=0.1, random_state=nRandomSeed)
        arrRawTrainData = arrRawData_df[arrRawData_df[ClassificationFeatures].isin(arrRawTrainDataUniqueClusters)]
        arrRawTrainDatalist = arrRawTrainData.index.tolist()
        arrRawTrainLabel = arrRawLabel[arrRawTrainDatalist]
        arrRawTrainData = np.array(arrRawTrainData).astype(np.float64)
        arrRawTestData = arrRawData_df[~arrRawData_df[ClassificationFeatures].isin(arrRawTrainDataUniqueClusters)]
        arrRawTestDatalist = arrRawTestData.index.tolist()
        arrRawTestLabel = arrRawLabel[arrRawTestDatalist]
        arrRawTestData = np.array(arrRawTestData).astype(np.float64)

    elif nSplitType == 3:
        print("\nSplitting method: by solvent")
        arrRawData = np.insert(arrRawData, arrRawData.shape[1], values=arrRawLabel, axis=1)
        arrRawData_df = pd.DataFrame(arrRawData, columns=np.append(arrRawFeatures, "DeltaGsolv"))
        ClassificationFeatures = "eps"
        arrRawGroupedData = arrRawData_df.groupby(arrRawData_df[ClassificationFeatures])
        ClassificationData = float(78.36)
        arrWaterData = arrRawGroupedData.get_group(ClassificationData)
        arrWaterDataIndices = arrWaterData.index
        arrRawData_df = arrRawData_df.drop(arrWaterDataIndices)
        arrWaterTrainData, arrWaterTestData = train_test_split(
            arrWaterData, test_size=0.1, random_state=nRandomSeed)
        arrRawDataUniqueClusters = arrRawData_df[ClassificationFeatures].unique()
        arrRawTrainDataUniqueClusters, _ = train_test_split(
            arrRawDataUniqueClusters, test_size=0.1, random_state=nRandomSeed)
        arrRawTrainData = arrRawData_df[arrRawData_df[ClassificationFeatures].isin(arrRawTrainDataUniqueClusters)]
        arrRawTrainData = pd.concat([arrWaterTrainData, arrRawTrainData])
        arrRawTestData = arrRawData_df[~arrRawData_df[ClassificationFeatures].isin(arrRawTrainDataUniqueClusters)]
        arrRawTestData = pd.concat([arrWaterTestData, arrRawTestData])
        arrRawTrainData = np.array(arrRawTrainData).astype(np.float64)
        arrRawTestData = np.array(arrRawTestData).astype(np.float64)
        np.random.seed(10)
        np.random.shuffle(arrRawTrainData)
        np.random.shuffle(arrRawTestData)
        nCol = arrRawData.shape[1] - 1
        arrRawTrainLabel = arrRawTrainData[:, nCol]
        arrRawTrainData = np.delete(arrRawTrainData, nCol, axis=1)
        arrRawTestLabel = arrRawTestData[:, nCol]
        arrRawTestData = np.delete(arrRawTestData, nCol, axis=1)

    np.save(f"{strPath}RawTrainData.npy", arrRawTrainData)
    np.save(f"{strPath}RawTrainLabel.npy", arrRawTrainLabel)
    np.save(f"{strPath}RawTestData.npy", arrRawTestData)
    np.save(f"{strPath}RawTestLabel.npy", arrRawTestLabel)

    bin_edges = np.arange(-21.0, 5.0, 0.5)
    num_bins = len(bin_edges)
    
    listTrainRatio = [0.0] * num_bins
    listTestRatio = [0.0] * num_bins
    
    for d in arrRawTrainLabel:
        index = int((d + 21.0) / 0.5)
        if 0 <= index < num_bins:
            listTrainRatio[index] += 1.0 / len(arrRawTrainLabel)
    
    for d in arrRawTestLabel:
        index = int((d + 21.0) / 0.5)
        if 0 <= index < num_bins:
            listTestRatio[index] += 1.0 / len(arrRawTestLabel)

    r, pValue = pearsonr(listTrainRatio, listTestRatio)
    print(f"Training size: {arrRawTrainData.shape[0]}")
    print(f"Test size: {arrRawTestData.shape[0]}")
    print(f"Pearson correlation between training and test label distribution: {math.fabs(r):.6f}")

    return arrRawTrainData, arrRawTrainLabel

def getValidFeatures(arrRawTrainData, arrRawTrainLabel, arrRawFeatures, strPath, strModel, strScoring):
    print("\nSelecting valid features...")

    nSelType = 2  # 1: PearsonMap + PearsonWithLabel, 2: PearsonMap + Forward, 3: Forward only

    arrSelTrainData = arrRawTrainData
    arrSelectFeatures = arrRawFeatures

    if nSelType == 1:
        arrOldSelectFeatures = arrSelectFeatures
        arrSelTrainData, arrSelectFeatures = featureSelection.selFeaturesByPearsonMap(
            arrSelTrainData, arrRawTrainLabel, arrSelectFeatures)
        listNotIn1, _ = method.compareTwoListDiff(arrSelectFeatures.tolist(), arrOldSelectFeatures)
        #print("\nFeatures removed after Pearson filter:", listNotIn1)

        arrOldSelectFeatures = arrSelectFeatures
        arrSelTrainData, arrSelectFeatures = featureSelection.selFeaturesByPearsonWithLabel(
            arrSelTrainData, arrRawTrainLabel, arrSelectFeatures)
        listNotIn1, _ = method.compareTwoListDiff(arrSelectFeatures.tolist(), arrOldSelectFeatures)
        #print("\nFeatures removed after PearsonWithLabel filter:", listNotIn1)

    elif nSelType == 2:
        arrOldSelectFeatures = arrSelectFeatures
        arrSelTrainData, arrSelectFeatures = featureSelection.selFeaturesByPearsonMap(
            arrSelTrainData, arrRawTrainLabel, arrSelectFeatures)
        listNotIn1, _ = method.compareTwoListDiff(arrSelectFeatures.tolist(), arrOldSelectFeatures)
        #print("\nFeatures removed after Pearson filter:", listNotIn1)

        arrOldSelectFeatures = arrSelectFeatures
        arrSelTrainData, arrSelectFeatures = featureSelection.selFeaturesForwardViaModel(
            arrSelTrainData, arrRawTrainLabel, arrSelectFeatures, strModel, strScoring)
        listNotIn1, _ = method.compareTwoListDiff(arrSelectFeatures.tolist(), arrOldSelectFeatures)
        #print("\nFeatures removed after forward selection:", listNotIn1)

    elif nSelType == 3:
        arrOldSelectFeatures = arrSelectFeatures
        arrSelTrainData, arrSelectFeatures = featureSelection.selFeaturesForwardViaModel(
            arrSelTrainData, arrRawTrainLabel, arrSelectFeatures, strModel, strScoring)
        listNotIn1, _ = method.compareTwoListDiff(arrSelectFeatures.tolist(), arrOldSelectFeatures)
        #print("\nFeatures removed after forward selection:", listNotIn1)

    print(f"\n{strModel}: Final selected features:", arrSelectFeatures)
    np.save(f"{strPath}UsedFeatures.npy", arrSelectFeatures)

    return arrSelTrainData, arrSelectFeatures

def trainModel(arrRawTrainData, arrRawTrainLabel, strModelName, strModelPath, curScoring):
    print("\nTraining model with scoring:", curScoring)

    bestNonlinerGrid = None

    if strModelName.find("SVM") > -1:
        kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-10, 1e10), nu=1.5)
        model = SVR(kernel=kernel, C=1.0, gamma=1.0, tol=dTol)
        param_grid = {'C': np.arange(1.0, 8.1, 0.1), 'gamma': np.logspace(-10, 0, 8)}

    elif strModelName.find("RF") > -1:
        model = RandomForestRegressor(n_estimators=100, max_depth=nMaxDepth, random_state=1, min_impurity_decrease=dEarlyStop)
        nFeatureNum = arrRawTrainData.shape[1]
        param_grid = {'n_estimators': np.arange(5, 101, 10), 'min_samples_leaf': [1], 'max_features': np.arange(1, nFeatureNum + 1)}

    elif strModelName.find("KRR") > -1:
        kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-10, 1e10), nu=1.5)
        model = KernelRidge(kernel=kernel, gamma=1.0)
        param_grid = {'alpha': np.arange(0.1, 2.1, 0.1), 'gamma': np.logspace(-10, 0, 10)}

    elif strModelName.find("GBRT") > -1:
        model = GradientBoostingRegressor(n_estimators=100, max_depth=nMaxDepth, min_impurity_decrease=dEarlyStop,
                                          random_state=10, learning_rate=0.2, alpha=0.9, criterion='squared_error',
                                          n_iter_no_change=10, loss='squared_error')
        nFeatureNum = arrRawTrainData.shape[1]
        param_grid = {'n_estimators': np.arange(5, 101, 10), 'min_samples_leaf': np.arange(1, 5),
                      'max_features': np.arange(1, nFeatureNum + 1), 'learning_rate': np.arange(0.1, 0.5, 0.1)}

    elif strModelName.find("XGBoost") > -1:
        model = XGBRegressor(n_estimators=100, max_depth=nMaxDepth, min_child_weight=5.0, gamma=5.0, random_state=10)
        param_grid = {'n_estimators': np.arange(5, 101, 10), 'learning_rate': np.arange(0.1, 0.5, 0.1),
                      'alpha': np.arange(0.01, 0.11, 0.01)}

    elif strModelName.find("GPR") > -1:
        kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-10, 1e10), nu=1.5)
        model = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, random_state=10)
        param_grid = {'alpha': np.logspace(0, 2, 10)}

    elif strModelName.find("LGBM") > -1:
        model = LGBMRegressor(boosting_type='dart', num_leaves=20, max_depth=nMaxDepth, learning_rate=0.1,
                              n_estimators=100, objective='regression', min_child_weight=1.0, random_state=10)
        param_grid = {'num_leaves': np.arange(5, 51, 5), 'n_estimators': np.arange(5, 101, 10),
                      'learning_rate': np.arange(0.1, 0.51, 0.1)}

    elif strModelName.find("ExRF") > -1:
        model = ExtraTreeRegressor(criterion='squared_error', splitter='best', max_depth=nMaxDepth,
                                   random_state=10, min_impurity_decrease=dEarlyStop)
        nFeatureNum = arrRawTrainData.shape[1]
        param_grid = {'min_samples_leaf': [1], 'max_features': np.arange(1, nFeatureNum + 1),
                      'min_samples_split': np.arange(2, 10)}

    else:
        raise ValueError(f"Unsupported model type: {strModelName}")

    bestNonlinerGrid = GridSearchCV(model, param_grid, scoring=curScoring, cv=10, n_jobs=-1)
    bestNonlinerGrid.fit(arrRawTrainData, arrRawTrainLabel)

    scores = cross_val_score(bestNonlinerGrid.best_estimator_, arrRawTrainData, arrRawTrainLabel, cv=10)
    print("Cross-validation scores:", scores)

    curModel = bestNonlinerGrid.best_estimator_
    joblib.dump(curModel, f"{strModelPath}{strModelName}-train.pkl")

    dictPara = curModel.get_params()
    print("Best parameters:", dictPara)

    return dictPara

def buildModel(arrRawTrainData, arrRawTrainLabel, strModelPath, strModelName, dictParas):
    print(f"\nBuilding model: {strModelName}")

    bestClassifer = None

    if strModelName.find("SVM") > -1:
        kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-10, 1e10), nu=1.5)
        bestClassifer = SVR(kernel=kernel, C=dictParas['C'], gamma=dictParas['gamma'], tol=dTol)

    elif strModelName.find("RF") > -1:
        bestClassifer = RandomForestRegressor(
            n_estimators=dictParas['n_estimators'],
            max_depth=dictParas['max_depth'],
            random_state=dictParas['random_state'],
            min_impurity_decrease=dEarlyStop,
            min_samples_leaf=dictParas['min_samples_leaf'],
            max_features=dictParas['max_features'])

    elif strModelName.find("GBRT") > -1:
        bestClassifer = GradientBoostingRegressor(
            n_estimators=dictParas['n_estimators'],
            max_depth=dictParas['max_depth'],
            min_impurity_decrease=dEarlyStop,
            random_state=dictParas['random_state'],
            loss=dictParas['loss'],
            learning_rate=dictParas['learning_rate'],
            alpha=dictParas['alpha'],
            max_features=dictParas['max_features'],
            min_samples_leaf=dictParas['min_samples_leaf'],
            criterion=dictParas['criterion'],
            n_iter_no_change=dictParas['n_iter_no_change'])

    elif strModelName.find("KRR") > -1:
        kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-10, 1e10), nu=1.5)
        bestClassifer = KernelRidge(kernel=kernel, alpha=dictParas['alpha'], gamma=dictParas['gamma'])

    elif strModelName.find("XGBoost") > -1:
        bestClassifer = XGBRegressor(
            n_estimators=dictParas['n_estimators'],
            max_depth=dictParas['max_depth'],
            min_child_weight=dictParas['min_child_weight'],
            gamma=dictParas['gamma'],
            random_state=dictParas['random_state'],
            learning_rate=dictParas['learning_rate'],
            alpha=dictParas['alpha'])

    elif strModelName.find("GPR") > -1:
        kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-10, 1e10), nu=1.5)
        bestClassifer = GaussianProcessRegressor(kernel=kernel, alpha=dictParas['alpha'], random_state=dictParas['random_state'])

    elif strModelName.find("LGBM") > -1:
        bestClassifer = LGBMRegressor(
            boosting_type='dart',
            num_leaves=dictParas['num_leaves'],
            max_depth=dictParas['max_depth'],
            learning_rate=dictParas['learning_rate'],
            n_estimators=dictParas['n_estimators'],
            objective='regression',
            min_child_weight=dictParas['min_child_weight'],
            random_state=dictParas['random_state'])

    bestClassifer.fit(arrRawTrainData, arrRawTrainLabel)

    model_file = f"{strModelPath}{strModelName}-best.pkl"
    joblib.dump(bestClassifer, model_file)
    print(f"Model saved to: {model_file}")

def testResultsEnsamble(arrRawFeatures, strSolvPath, strTrain):
    print("\nEnsemble Testing Start...")

    if strTrain == "Train":
        print("== Using training set for prediction ==")
        arrRawTestData = np.load(f"{strSolvPath}RawTrainData.npy")
        arrRawTestLabel = np.load(f"{strSolvPath}RawTrainLabel.npy")
        outFileName = BASISDIR + "train_predictions.csv"

    elif strTrain == "Test":
        print("== Using test set for prediction ==")
        arrRawTestData = np.load(f"{strSolvPath}RawTestData.npy")
        arrRawTestLabel = np.load(f"{strSolvPath}RawTestLabel.npy")
        outFileName = BASISDIR + "test_predictions.csv"

    elif strTrain == "Raw":
        print("== Using raw CSV file for prediction ==")
        arrRawData = np.genfromtxt(BASISDIR + "your_rawfile.csv", dtype=str, delimiter=",")
        nCol = arrRawData.shape[1] - 1
        arrRawFeatures = arrRawData[0, 0:nCol]
        arrRawData = np.delete(arrRawData, 0, axis=0)
        arrRawTestLabel = arrRawData[:, nCol].astype(np.float64)
        arrRawTestData = np.delete(arrRawData, nCol, axis=1).astype(np.float64)
        outFileName = BASISDIR + "raw_predictions.csv"

    scaler = joblib.load(f"{strSolvPath}MinMaxScaler.pkl")
    arrRawTestData = scaler.transform(arrRawTestData)

    arrVarSelectFeatures = np.load(f"{strSolvPath}FeatureAfterVar.npy")
    for nSubIndex in range(len(arrRawFeatures)-1, -1, -1):
        if arrRawFeatures[nSubIndex] not in arrVarSelectFeatures:
            arrRawTestData = np.delete(arrRawTestData, nSubIndex, axis=1)

    scaler = joblib.load(f"{strSolvPath}StandardScaler.pkl")
    arrRawTestData = scaler.transform(arrRawTestData)

    listModleName = ["LGBM", "RF", "GBRT", "XGBoost"]
    listAllResults = []
    strModelNames = ""

    for strModelName in listModleName:
        strModelPath = f"{strSolvPath}{strModelName}/"
        strModelFile = f"{strModelPath}/{strModelName}-best.pkl"

        if not os.path.exists(strModelFile):
            continue

        print(f"Running model: {strModelName}")
        strModelNames += f"{strModelName}-"

        arrSelSelectFeatures = np.load(f"{strModelPath}UsedFeatures.npy")
        arrModelRawTestData = arrRawTestData.copy()

        for nSubIndex in range(len(arrVarSelectFeatures)-1, -1, -1):
            if arrVarSelectFeatures[nSubIndex] not in arrSelSelectFeatures:
                arrModelRawTestData = np.delete(arrModelRawTestData, nSubIndex, axis=1)

        curModel = joblib.load(strModelFile)
        predictions = curModel.predict(arrModelRawTestData)
        listAllResults.append(predictions)

        R2 = r2_score(arrRawTestLabel, predictions)
        MAE = mean_absolute_error(arrRawTestLabel, predictions)
        RMSE = math.sqrt(mean_squared_error(arrRawTestLabel, predictions))
        print(f"{strModelName}\tR2: {R2:.4f}\tMAE: {MAE:.4f}\tRMSE: {RMSE:.4f}")

    # Average ensemble prediction
    nTestNum = arrRawTestData.shape[0]
    listFinalResult = np.mean(np.array(listAllResults), axis=0)

    R2 = r2_score(arrRawTestLabel, listFinalResult)
    MAE = mean_absolute_error(arrRawTestLabel, listFinalResult)
    RMSE = math.sqrt(mean_squared_error(arrRawTestLabel, listFinalResult))
    print(f"{strModelNames}\tEnsemble R2: {R2:.4f}\tMAE: {MAE:.4f}\tRMSE: {RMSE:.4f}")

    # Save predictions
    with open(outFileName, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["True", "Predicted"])
        for true_val, pred_val in zip(arrRawTestLabel, listFinalResult):
            writer.writerow([true_val, pred_val])
    print(f"Predictions saved to: {outFileName}")

def showResult():

    strSolvPath = "%sML-Out/%s/"%(BASISDIR,str("Solv"))
    strInitialFeatureFile = strSolvPath + "InitialFeature.npy"
    arrRawFeatures = np.load(strInitialFeatureFile)

    testResultsEnsamble(arrRawFeatures,strSolvPath,"Train")

    testResultsEnsamble(arrRawFeatures,strSolvPath, "Test")

    #testResultsEnsamble(arrRawFeatures,strSolvPath, "Raw")

if __name__ == '__main__':
    listModleName = ["LGBM", "RF", "GBRT", "XGBoost"]  
    curScoring = "neg_root_mean_squared_error"

    strSolvPath = f"{BASISDIR}ML-Out/Solv/"
    os.makedirs(strSolvPath, exist_ok=True)

    strOutFile = BASISDIR + "Result.out"
    outFile = open(strOutFile, "w")
    oldStdOut = sys.stdout
    sys.stdout = outFile

    print("\nTraining Start...")
    print("Start time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    arrRawData, arrRawLabel, arrRawFeatures = getIntitialDataSet(strSolvPath)
    arrRawTrainData, arrRawTrainLabel = splitDataSet(arrRawData, arrRawLabel, arrRawFeatures, strSolvPath)

    scaler = MinMaxScaler().fit(arrRawTrainData)
    arrRawTrainData = scaler.transform(arrRawTrainData)
    joblib.dump(scaler, f"{strSolvPath}MinMaxScaler.pkl")

    arrVarSelTrainData, arrVarSelectFeatures = featureSelection.selFeaturesByVariance(arrRawTrainData, arrRawFeatures)
    listNotIn1, _ = method.compareTwoListDiff(arrVarSelectFeatures.tolist(), arrRawFeatures.tolist())
    print("\nFeatures removed after initial variance filtering:", listNotIn1)
    np.save(f"{strSolvPath}FeatureAfterVar.npy", arrVarSelectFeatures)

    standScaler = StandardScaler().fit(arrVarSelTrainData)
    arrVarSelTrainData = standScaler.transform(arrVarSelTrainData)
    joblib.dump(standScaler, f"{strSolvPath}StandardScaler.pkl")

    for strModelName in listModleName:
        print(f"\nTraining model: {strModelName}")
        strModelPath = f"{strSolvPath}/{strModelName}/"
        os.makedirs(strModelPath, exist_ok=True)

        arrModelSelTrainData, arrModelSelectFeatures = getValidFeatures(
            arrVarSelTrainData, arrRawTrainLabel, arrVarSelectFeatures,
            strModelPath, strModelName, curScoring)

        dictDictPara = trainModel(arrModelSelTrainData, arrRawTrainLabel, strModelName, strModelPath, curScoring)
        buildModel(arrModelSelTrainData, arrRawTrainLabel, strModelPath, strModelName, dictDictPara)

    print("\nEnd time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("Training finished.\n")

    showResult()
    outFile.close()
    sys.stdout = oldStdOut

