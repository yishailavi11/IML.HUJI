import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
from datetime import datetime
from matplotlib import pyplot as plt


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename,parse_dates=True)
    df = df.query('Temp > -20').dropna()
    day_of_year_func = lambda x: int(datetime.strptime(x,'%Y-%m-%d').strftime('%j'))
    df['DayOfYear'] = df['Date'].apply(day_of_year_func)
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data('../datasets/City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    # first plot
    idf = df.query('Country == "Israel"')
    plt.figure()
    cmap = plt.get_cmap('jet', 20)
    cmap.set_under('gray')
    sc = plt.scatter(idf.DayOfYear.values, idf.Temp.values, s=5, c=idf.Year.values, cmap=cmap)
    plt.colorbar(sc)
    plt.ylabel('Temp')
    plt.xlabel('Day of year')
    plt.title('Avarage temperature in TV per day in the years 95-07')
    plt.show()
    
    #second plot
    months = np.arange(1,13,1)
    plt.figure()
    plt.bar(months, idf.groupby('Month').agg('std').Temp, tick_label=months)
    plt.xlabel('std')
    plt.ylabel('Month')
    plt.title('Standard deviation of daily temperature in each month')
    plt.show()

    # Question 3 - Exploring differences between countries
    groups = df.groupby(['Country','Month']).Temp.agg(['mean','std'])

    plt.figure(figsize=(8,6))
    for i, (country, data) in enumerate(groups.groupby(level=0)):
        plt.errorbar(months,data['mean'].values,data['std'].values,\
                     label=country,capsize=(i+1)*2)
    plt.legend(loc='lower right')
    plt.xticks(months)
    plt.xlabel('Month')
    plt.ylabel('Average temperature')
    plt.title('Average temp per month with error bars')
    plt.show()

    # Question 4 - Fitting model for different values of `k`
    X, y = idf.DayOfYear, idf.Temp
    train_X, train_y, test_X, test_y = split_train_test(X,y,0.75)
    loss = np.zeros(10)
    for i in range(10):
        polyfit = PolynomialFitting(i+1)
        polyfit.fit(train_X,train_y)
        loss[i] = polyfit.loss(test_X, test_y)
    print(loss)

    plt.figure()
    plt.bar(np.arange(1,11),loss)
    plt.xticks(np.arange(1,11))
    plt.xlabel('k')
    plt.ylabel('loss')
    plt.title('Loss of the polyfit as a function of k')    
    plt.show()

    # Question 5 - Evaluating fitted model on different countries
    ipolyfit = PolynomialFitting(5)
    X,y = idf.DayOfYear, idf.Temp
    ipolyfit.fit(X,y)

    countries = ['The Netherlands', 'Jordan', 'South Africa']
    loss = np.zeros(3)
    for i, country in enumerate(countries):
        cur_df = df.query('Country == @country')
        cur_X, cur_y = cur_df.DayOfYear, cur_df.Temp
        loss[i] = ipolyfit.loss(cur_X, cur_y)
        
    plt.bar(countries, loss)
    plt.xlabel('Country')
    plt.ylabel('loss')
    plt.title('Loss value of fitted polynome on different countries')
    plt.show()