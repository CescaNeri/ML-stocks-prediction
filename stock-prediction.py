# Import Packages and Libraries

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from finta import TA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from numpy.core.arrayprint import format_float_scientific
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score, confusion_matrix

# Set functions

def get_data(stock):
    return yf.Ticker(stock).history(period='max')

def clean(df, del_list):
  for item in del_list:
    del df[item]
  return df

def target(df, t_name, target_name):
  df[t_name] = df['Close'].shift(-1)
  df[target_name] = (df[t_name] > df['Close']).astype(int)
  return df

def accuracy(model, test, predictors):
  preds = model.predict(test[predictors])
  preds = pd.Series(preds, index = test.index)
  solution = precision_score(test['Target'], preds)
  print('Final accuracy amounts to: {}'.format(solution))
  combined = pd.concat({'Target': test['Target'],'Predictions': preds}, axis=1)
  combined.plot()

def random_forest(df, predictors, split):
  df = df.dropna()
  model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
  train = df.iloc[:split]
  test = df.iloc[split:]
  model.fit(train[predictors], train['Target'])
  return accuracy(model, test, predictors)

def smooth_data(data, alpha):
  return data.ewm(alpha=alpha).mean()

def tech_analysis(df, indicators):
  df.rename(columns = {'Close': 'close', 'High': 'high', 'Low': 'low', 
                       'Adj Close': 'adj close', 'Volume': 'volume', 'Open': 'open'}, 
            inplace = True)
  for indicator in indicators:
    df[indicator] = eval('TA.' + indicator + '(data)')
  return df

# Execute the Model

model_choice = input("""Select a Machine Learning model among these three: 
    \n 1. Bitcoin Model (type: 1) 
    \n 2. Financial Indicators Model (type: 2) 
    \n 3. Regression Model (type: 3) 
    \n""")

if model_choice == '1':
  nvidia = get_data('NVDA')
  bitcoin = get_data('BTC')

  to_del = ['Dividends', 'Stock Splits']
  clean(nvidia, to_del)
  clean(bitcoin, to_del)

  target(nvidia, 't1', 'Target')
  target(bitcoin, 't1_b', 'Target_b')

  bitcoin.rename(columns = {'Close': 'Close_B', 'High': 'High_B', 'Low': 'Low_B', 
                       'Adj Close': 'Adj Close_B', 'Volume': 'Volume_B', 'Open': 'Open_B'}, 
            inplace = True)
  nvidia = nvidia.merge(bitcoin, how='inner', on='Date')

  predictors = ['Open', 'High', 'Low', 'Close', 'Volume', 't1', 'Open_B',
       'High_B', 'Low_B', 'Close_B', 'Volume_B', 't1_b', 'Target_b']
  random_forest(nvidia, predictors, -50)

elif model_choice == '2':
  nvidia = get_data('NVDA')

  to_del = ['Dividends', 'Stock Splits']
  clean(nvidia, to_del)
  target(nvidia, 't1', 'Target')

  alpha = 0.65
  data = smooth_data(nvidia, alpha)
  indicators = ['OBV', 'ADL', 'ADX', 'RSI', 'STOCH', 'SMA']
  tech_analysis(data, indicators)

  to_del = ['open', 'high', 'low', 'close', 'volume', 't1', 'Target']
  clean(data, to_del)
  nvidia = nvidia.merge(data, how='inner', on='Date')

  predictors = ['Open', 'High', 'Low', 'Close', 'Volume', 't1', 'OBV', 'ADL',
       'ADX', 'RSI', 'STOCH', 'SMA']
  random_forest(nvidia, predictors, -50)

