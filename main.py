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

    df.hist(column=['LotFrontage'])
    plt.title('LotFrontage')
    plt.show()
    df.plot(type='Bar', column=['LotFrontage'])
    plt.title('LotFrontage')
    plt.show()

    # Q1 = df['YearBuilt'].quantile(0.25)
    # Q3 = df['YearBuilt'].quantile(0.75)
    # IQR = Q3 - Q1  # IQR is interquartile range.
    #
    # upperFilter = (df['YearBuilt'] >= Q3 + 1.5 * IQR)
    # lowerFilter = (df['YearBuilt'] <= Q1 - 1.5 * IQR)
    # df.loc[upperFilter, ['YearBuilt']] = Q3 + 1.5 * IQR
    # df.loc[lowerFilter, ['YearBuilt']] = Q1 - 1.5 * IQR
    # df.boxplot(column=['YearBuilt'])
    # df.hist(column=['YearBuilt'])
    # plt.title('YearBuilt after removing outliers')
    # plt.show()
    #
    # #
    # standartizedDf = (df['YearBuilt'] - df['YearBuilt'].mean()) / df['YearBuilt'].std()
    # standartizedDf.hist()
    # plt.title('YearBuilt Standartized')
    # plt.show()
    #
    # normalizedDf = (df['YearBuilt'] - df['YearBuilt'].min()) / (df['YearBuilt'].max() - df['YearBuilt'].min())
    # normalizedDf.hist()
    # plt.title('YearBuilt Normalized')
    # plt.show()
    #
    # # show frequency of each value in column figure
    # counts = showFrequency(df, 'SaleCondition')
    # counts.plot(kind='pie', autopct='%1.0f%%')
    # plt.title('SaleCondition Frequency')
    # plt.show()