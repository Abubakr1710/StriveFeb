# Importng needed libraries
import numpy as np
import pandas as pd
import pip

from sklearn.model_selection import train_test_split

from sklearn.ensemble      import GradientBoostingRegressor
from xgboost               import XGBRegressor
from lightgbm              import LGBMRegressor
from sklearn.linear_model  import LinearRegression

import time
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
#----------------------------------------------------------------------------------#
df = pd.read_csv('Chapter 02/16. TimeSeries/climate.csv')
df = df.drop(columns="Date Time")

def get_sequence(data, seq_len, target_name):

    seq_list = []
    target_list = []

    for i in range(0, data.shape[0] - (seq_len+1), seq_len+1):

        seq = data[i: seq_len + i]
        target = data[target_name][seq_len + i]

        seq_list.append(seq)
        target_list.append(target)

    return np.array(seq_list), np.array(target_list)

x, y = get_sequence(df, seq_len= 6, target_name='T (degC)')

print(x.shape)
print(y.shape)
#----------------------------------------------------------------------------------#

def get_features(x):
    feature = []
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            mean_column = np.mean(x[i, j, :])


        feature.append(mean_column)
        #feature =np.hstack((mean_column_1, std_columns_2))
    return np.array(feature)

nx = get_features(x)
print(nx.shape)