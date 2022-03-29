from urllib import response
from pyparsing import col
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
from os.path import join
from matplotlib import pyplot as plt


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv('/home/yishailavi124/IML.HUJI/datasets/house_prices.csv')
    df.date = pd.to_datetime(df.date, errors='coerce').astype(int)
    df = df.drop(columns=['id','date'])
    df = df.dropna()

    positive_only = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
                     'floors', 'condition', 'grade', 'sqft_above', 'price',
                     'sqft_living15', 'sqft_lot15','yr_built']
    positive_query = ''
    for feature in positive_only:
        positive_query += '(' + feature + '>0)&'
    positive_query = positive_query[:-1]
    df = df.query(positive_query)   
    
    y = df.price
    X = df.drop(columns='price')
    return X, y


def calc_pearson(x: pd.Series, y: pd.Series) -> float:
    """
    Calculates Pearson Correlation between two features.
    
    Parameters
    ----------
    x : array-like of shape (n_samples, )
        Feature vector
    
    y : array-like of shape (n_samples, )
        Response vector

    Returns
    -------
    Pearson Correlation between x,y.
    """
    x, y = x.values, y.values
    return np.cov(x, y)[0][1]/(np.std(x) * np.std(y))


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for i, feature in enumerate(list(X.columns)):
        plt.figure(i)
        plt.xlabel(feature)
        plt.ylabel('price')
        plt.title('Pearson Correlation: ' + str(calc_pearson(X[feature], y)))
        plt.scatter(X[feature].values, y.values, s=3)
        plt.savefig(join(output_path, feature))
        

if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data('../datasets/house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, './exercises/plots/')

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X,y,0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    range_arr = np.arange(10,101,1)
    mean_list = []
    std_list = []
    for p in range_arr:
        p_loss = np.zeros(10)
        for i in range(10):
            sample = train_X.sample(frac=p/100)
            reg_model = LinearRegression()
            reg_model.fit(sample, train_y[sample.index])
            p_loss[i] = reg_model.loss(test_X, test_y)
        mean_list.append(p_loss.mean())
        std_list.append(p_loss.std())
    mean_list = np.asarray(mean_list)    
    std_list = np.asarray(std_list) 
    confidence = 2*std_list

    plt.figure()
    plt.plot(range_arr,mean_list)
    plt.fill_between(range_arr, mean_list-confidence, mean_list+confidence, color='b', alpha=.1)
