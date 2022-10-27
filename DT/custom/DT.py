# using decission tree predict prices
import time

import numpy as np
import pandas as pd
from DecisionTree import trainTestSplit, buildDecisionTree, decisionTreePredictions, calculateAccuracy


def main():
    df = pd.read_csv("../../historicalData.tsv", sep='\t')
    df.drop(['Id'], axis=1, inplace=True)
    continuousFeatures = df.select_dtypes(exclude=['object']).columns.tolist()

    # get LotFrontage mean

    for column in continuousFeatures:
        mean = df[column].mean()
        df[column] = fixNAValuesOfColumns(df[column], mean)
    # df = fixNAValuesOfColumns(df)
    streets = createMapForColumn(df["Street"])
    df['Street'] = df['Street'].map(streets)

    # get all Neighbourhoods
    neighbourhoods = createMapForColumn(df['Neighborhood'])
    df['Neighborhood'] = df['Neighborhood'].map(neighbourhoods)

    centralAir = createMapForColumn(df['CentralAir'])
    df['CentralAir'] = df['CentralAir'].map(centralAir)

    pavedDrive = createMapForColumn(df['PavedDrive'])
    df['PavedDrive'] = df['PavedDrive'].map(pavedDrive)

    saleCondition = createMapForColumn(df['SaleCondition'])
    df['SaleCondition'] = df['SaleCondition'].map(saleCondition)

    dataFrameTrain, dataFrameTest = trainTestSplit(df, testSize=0.7)

    # dataFrameTest.drop(['SalePrice'], axis=1, inplace=True)
    # dataFrameTrain.drop(['SalePrice'], axis=1, inplace=True)

    print("Decision Tree - House Prices Dataset")
    decisionTree = 0
    maxDepth = 1000
    minSamplesLeaf = 2
    accuracyTrain = 0
    accuracyTest = 0
    buildingTime = 0
    startTime = time.time()
    while accuracyTrain < 98:
        decisionTree = buildDecisionTree(dataFrameTrain.values, dataFrameTrain.columns, maxDepth=maxDepth, minSampleLeaf=minSamplesLeaf)
        buildingTime = time.time() - startTime
        decisionTreeTestResults = decisionTreePredictions(dataFrameTest, decisionTree)
        accuracyTest = calculateAccuracy(decisionTreeTestResults, dataFrameTest.iloc[:, -1]) * 100
        decisionTreeTrainResults = decisionTreePredictions(dataFrameTrain, decisionTree)
        accuracyTrain = calculateAccuracy(decisionTreeTrainResults, dataFrameTrain.iloc[:, -1]) * 100
        maxDepth += 1

    print("maxDepth = {}: ".format(maxDepth), end="")
    print("accTest = {0:.2f}%, ".format(accuracyTest), end="")
    print("accTrain = {0:.2f}%, ".format(accuracyTrain), end="")
    print("buildTime = {0:.2f}s".format(buildingTime), end="\n")

    data = pd.DataFrame(np.array([[65, 8450, 0, 0, 2003, 2003, 0,0,0]], dtype=object),
                        columns=['LotFrontage','LotArea', 'Street', 'Neighborhood', 'YearBuilt', 'YearRemodAdd', 'CentralAir', 'PavedDrive', 'SaleCondition'])
    predict = decisionTreePredictions(data, decisionTree)
    print('Predicted price: ', predict[0])

    df = df[['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'SalePrice']]


def createMapForColumn(column):
    values = column.unique()
    d = {}
    for i in range(len(values)):
        d[values[i]] = i
    return d


def fixNAValuesOfColumns(column, mean):
    column = column.fillna(mean)
    return column


main()
