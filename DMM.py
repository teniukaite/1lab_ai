import pandas as pd

if __name__ == '__main__':
    initialData = pd.read_csv("historicalData.tsv", sep='\t')
    initialTargetValues = initialData['SalePrice']
    k = 5
    minSampleLeaf = 2
    maxDepth = 30
    nEstimators = 100


