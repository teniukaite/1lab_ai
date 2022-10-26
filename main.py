# importing necessary librarys
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset


# show frequency of each value in column
def showFrequency(df, column):
    counts = df[column].value_counts()

    return counts


if __name__ == '__main__':
    df = pd.read_csv("historicalData.tsv", sep='\t')
    df.head()

    # df.hist(column=['LotArea'])
    # plt.title('LotArea')
    # plt.show()
    # df.plot(type='Bar', column=['LotArea'])
    # plt.title('LotArea')
    # plt.show()

    Q1 = df['LotArea'].quantile(0.25)
    Q3 = df['LotArea'].quantile(0.75)
    IQR = Q3 - Q1  # IQR is interquartile range.

    upperFilter = (df['LotArea'] >= Q3 + 1.5 * IQR)
    lowerFilter = (df['LotArea'] <= Q1 - 1.5 * IQR)
    df.loc[upperFilter, ['LotArea']] = Q3 + 1.5 * IQR
    df.loc[lowerFilter, ['LotArea']] = Q1 - 1.5 * IQR
    df.boxplot(column=['LotArea'])
    df.hist(column=['LotArea'])
    plt.title('LotArea after removing outliers')
    plt.show()

    #
    standartizedDf = (df['LotArea'] - df['LotArea'].mean()) / df['LotArea'].std()
    standartizedDf.hist()
    plt.title('LotArea Standartized')
    plt.show()

    normalizedDf = (df['LotArea'] - df['LotArea'].min()) / (df['LotArea'].max() - df['LotArea'].min())
    normalizedDf.hist()
    plt.title('LotArea Normalized')
    plt.show()

    # show frequency of each value in column figure
    counts = showFrequency(df, 'SaleCondition')
    counts.plot(kind='pie', autopct='%1.0f%%')
    plt.title('SaleCondition Frequency')
    plt.show()