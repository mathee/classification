"""functions that are useful for exploratory analysis of data"""
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# correlation matrix with seaborn

def correlation_matrix(df):
    '''prints a annotated correlation matrix of the given dataset
    IN: {df: pd.DataFrame}'''
    corrmat = df.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True, annot=True)
    
def get_k_most_correlated(df, y_column_label, k):
    '''returns k most correlated columns in given dataset as list
    IN: {df: pd.DataFrame, y_column_label: string, k: integer}
    OUT: {cols: list}
    '''
    corrmat = df.corr()
    cols = corrmat.nlargest(k, y_column_label)[y_column_label].index
    return cols

def scatter_plot(df, col_label1, col_label2):
    '''shows scatter plots of the given input.
    IN: {df:pd.DataFrame, col_label1 & 2: strings}'''
    data = pd.concat([df[col_label1], df[col_label2]], axis=1)
    data.plot.scatter(x=col_label2, y=col_label1, ylim=(0,800000))

def scatter_plots(df, cols):
    '''shows many scatter plots of the given input at a time
    IN: {df:pd.DataFrame, cols:list of column_labels}'''
    sns.set()
    sns.pairplot(df[cols], height = 2.5)
    plt.show()
    
def show_missing_data(df, n_most_columns):
    '''returns a tabular report on the missing data within dataset
    IN: {df:pd.DataFrame, n_most_columns:integer}
    OUT: {missing_data: pd.DataFrame}'''
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data.head(n_most_columns))

def show_balance(df, y_column_label):
    '''check visually if the dataset is balanced'''
    sns.catplot(y=y_column_label, kind="count", data=df, height=2.6, aspect=2.5, orient='h')