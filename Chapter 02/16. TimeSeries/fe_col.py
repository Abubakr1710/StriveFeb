# Importng needed libraries
import numpy as np
import pandas as pd

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
        mean_column = x[i].mean(axis=0)
        feature.append(mean_column)
    return np.array(feature)

nx = get_features(x)
#print(nx.shape)

def split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split(nx, y)

#----------------------------------------------------------------------------------------------------------------------------------------#  

def tree_regressors():
    tree_regressor = {
    'Linear':        LinearRegression(),
    "Skl GBM":       GradientBoostingRegressor(),
    "XGBoost":       XGBRegressor(),
    "LightGBM":      LGBMRegressor()
    }
    tree_regressor ={model_name: Pipeline([('scaler', StandardScaler()),('model', model)])for model_name, model in tree_regressor.items()}
    return tree_regressor

reg =tree_regressors()
#------------------------------------------------------------------------------#

results = pd.DataFrame({'Model': [], 'MSE': [], 'MAB': [], 'R2_score': [],'Time': [],})
for model_name, model in reg.items():

    start_time = time.time()
    model.fit(X_train, y_train)
    total_time = time.time() - start_time
        
    pred = model.predict(X_test)
    
    results = results.append({"Model":    model_name,
                              "MSE": metrics.mean_squared_error(y_test, pred),
                              "MAB": metrics.mean_absolute_error(y_test, pred),
                              "R2_score": metrics.r2_score(y_test, pred),
                              "Time":     total_time
                              },
                              ignore_index=True)



results_ord = results.sort_values(by=['MSE'], ascending=True, ignore_index=True)
results_ord.index += 1 
results_ord.style.bar(subset=['MSE', 'MAE'], vmin=0, vmax=100, color='#5fba7d')

print(results_ord)