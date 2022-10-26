import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier


# create confusion matrix
def confusion_matrix(y_true, y_pred):
    # get unique labels
    labels = np.unique(np.concatenate((y_true, y_pred), axis=0))
    # create matrix
    matrix = np.zeros((len(labels), len(labels)))
    # fill matrix
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            matrix[i, j] = np.sum(np.logical_and(y_true == label_i, y_pred == label_j))

    #coun missclassification rate
    missclassification_rate = np.sum(np.logical_and(y_true != y_pred, y_true != 0)) / len(y_true)

    #count classification accuracy
    classification_accuracy = np.sum(np.logical_and(y_true == y_pred, y_true != 0)) / len(y_true)

    return matrix, missclassification_rate, classification_accuracy

# count MAE
def MAE(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / len(y_true)

# count MAPE
def MAPE(y_true, y_pred):
    return np.sum(np.abs((y_true - y_pred) / y_true)) / len(y_true)


if __name__ == '__main__':
    df = pd.read_csv("../../historicalData.tsv", sep='\t')
    testData = pd.read_csv("../../historicalData.tsv", sep='\t')
    testData = testData.drop(['SalePrice', 'Id'], axis=1)
    trainingData = df.drop(['SalePrice', 'Id'], axis=1)
    le = preprocessing.LabelEncoder()
    testData = testData.apply(le.fit_transform)
    transformed = trainingData.apply(le.fit_transform)
    minSampleLeaf = 2
    maxDepth = 30
    nEstimators = 100

    clf = RandomForestClassifier(n_estimators=nEstimators, min_samples_leaf=minSampleLeaf, max_depth=maxDepth)
    clf.fit(transformed, df['SalePrice'])

    query = pd.DataFrame(np.array([[65, 8450, 0, 0, 2003, 2003, 0, 0, 0]], dtype=object),
                            columns=['LotFrontage', 'LotArea', 'Street', 'Neighborhood', 'YearBuilt', 'YearRemodAdd',
                                        'CentralAir', 'PavedDrive', 'SaleCondition'])
    predicted = clf.predict(testData)
    # print(predicted)
    matrix, missclassification_rate, classification_accuracy = confusion_matrix(df['SalePrice'], clf.predict(transformed))
    print('confusion matrix:')
    print(matrix)
    print('missclassification rate:')
    print(missclassification_rate)
    print('classification accuracy:')
    print(classification_accuracy)