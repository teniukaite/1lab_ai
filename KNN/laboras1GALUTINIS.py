import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd


def main():
    df = pd.read_csv("../historicalData.tsv", sep='\t')
    df = df[['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'SalePrice', 'CentralAir']]  # Columns which will be selected

    # Standartization of data for continous variables
    df = fixNAValuesOfColumns(df)
    data = df.copy()
    # preparedData = convertCategoricToBoolean(data)
    preparedData = prepareData(data, df)
    convertToNumeric(preparedData, 'CentralAir')
    salePrice_column = preparedData["SalePrice"]
    preparedData = preparedData.drop(columns=["SalePrice"])
    preparedData = pd.concat([preparedData, salePrice_column], axis=1)

    query, sortedDistances, nearestIndices = calculateNeighbours(preparedData, 8, df)

    indices = list(zip(*nearestIndices))[1]

    nearest = df.loc[indices,:]  # Nearest neighbours

    print("Predicted price for ")
    print(query)
    print("predicted price: {0:7.2f}".format(nearest["SalePrice"].mean()))

def dataPreprocessing(updatedData, initialData, column):
    if column.isnumeric():
        updatedData[column] = (updatedData[column] - initialData[column].mean()) / initialData[column].std()


def calculateNeighbours(data, k, df):
    query = pd.DataFrame(np.array([[57, 7449, 1930, 1950, 1]]),
                         columns=['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'CentralAir'])

    preparedQuery = query.copy()
    # preparedQuery = convertCategoricToBoolean(preparedQuery)
    preparedQuery = prepareData(preparedQuery, df)

    distances = []
    for i in range(np.shape(data)[0]):
        # use  Euclidean distance
        distance = np.linalg.norm(data.iloc[[i], :-1].to_numpy() - preparedQuery.iloc[[0]].to_numpy())
        distances.append((distance, i))

    sortedDistances = sorted(distances)
    kNearestIndices = sortedDistances[:k]

    return query, sortedDistances, kNearestIndices


def fixNAValuesOfColumns(df):
    for column in df.columns:
        if df[column].isnull().values.any().sum() * 100 / df[column].count() < 60:
            for index, row in df.iterrows():
                if pd.isnull(row[column]):
                    df[column].fillna(df[column].mean(), inplace=True)
        else:
            df.drop(column, axis=1, inplace=True)

    return df

def prepareData(data, df):
    for (columnName, columnData) in data.items():
        dataPreprocessing(data, df, columnName)

    return data

def calculateDistance(row, query):
    distance = 0
    for index, value in row.items():
        distance += (value - query[index]) ** 2

    return distance ** 0.5

def salePriceMean(data):
    mean = 0
    for row in data:
        for (columnName, columnData) in row.items():
            if columnName == 'SalePrice':
                mean = mean + int(columnData)

    return mean / len(data)

#calculate Hamming distance
def hammingDistance(row, query):
    distance = 0
    for index, value in row.items():
        if value != query[index]:
            distance += 1

    return distance

def convertToNumeric(data, column):
    map = {'Y': 1, 'N': 0}
    data[column] = data[column].map(map)

main()
