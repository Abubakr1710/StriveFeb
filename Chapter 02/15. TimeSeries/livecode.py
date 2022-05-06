from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble      import GradientBoostingRegressor
from xgboost               import XGBRegressor
from lightgbm              import LGBMRegressor
from sklearn.linear_model  import LinearRegression

import time
from sklearn import metrics


df = pd.read_csv('Chapter 02/15. TimeSeries/climate.csv')
df = df.drop(['Date Time'], axis=1)

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

def get_feat(x):
    nx =[]
    for i in range(x.shape[0]):
        meanx = x[i].mean(axis=0)
        stdx =x[i].std(axis=0)
        nx.append((meanx, stdx))
    
    return np.array(nx)
nx =get_feat(x)
print(nx.shape)

def splitting(nx,y):
    X_train, X_test, y_train, y_test = train_test_split(nx, y, test_size=0.2, random_state=0)
    return X_train, X_test ,y_train, y_test

X_train, X_test ,y_train, y_test =splitting(nx,y)


def tree_regressors():
    tree_regressor = {
    'Linear':        LinearRegression(),
    "Skl GBM":       GradientBoostingRegressor(n_estimators=100),
    "XGBoost":       XGBRegressor(n_estimators=100),
    "LightGBM":      LGBMRegressor(n_estimators=100)
    }
    return tree_regressor

reg =tree_regressors()
results = pd.DataFrame({'Model': [], 'MSE': [], 'MAB': [], 'Time': [],})
for model_name, model in reg.items():

    start_time = time.time()
    model.fit(X_train, y_train)
    total_time = time.time() - start_time
        
    pred = model.predict(X_test)
    
    results = results.append({"Model":    model_name,
                              "MSE": metrics.mean_squared_error(y_test, pred),
                              "MAB": metrics.mean_absolute_error(y_test, pred),
                              "Time":     total_time
                              },
                              ignore_index=True)



results_ord = results.sort_values(by=['MSE'], ascending=True, ignore_index=True)
results_ord.index += 1 
results_ord.style.bar(subset=['MSE', 'MAE'], vmin=0, vmax=100, color='#5fba7d')

print(results_ord)