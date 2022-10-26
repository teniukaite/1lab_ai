import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier


if __name__ == '__main__':
    df = pd.read_csv("../../historicalData.tsv", sep='\t')
    trainingData = df.drop(['SalePrice', 'Id'], axis=1)
    le = preprocessing.LabelEncoder()
    transformed = trainingData.apply(le.fit_transform)
    print(transformed)
    minSampleLeaf = 2
    maxDepth = 30

    clf = DecisionTreeClassifier(min_samples_leaf=minSampleLeaf, max_depth=maxDepth)
    clf.fit(transformed, df['SalePrice'])

    query = pd.DataFrame(np.array([[65, 8450, 0, 0, 2003, 2003, 0, 0, 0]], dtype=object),
                            columns=['LotFrontage', 'LotArea', 'Street', 'Neighborhood', 'YearBuilt', 'YearRemodAdd',
                                        'CentralAir', 'PavedDrive', 'SaleCondition'])
    predicted = clf.predict(query)
    print(predicted)
