from pyexpat import model
from sklearn.impute import  SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn import pipeline
from sklearn import compose
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd


from sklearn.tree          import DecisionTreeRegressor
from sklearn.ensemble      import RandomForestRegressor
from sklearn.ensemble      import ExtraTreesRegressor
from sklearn.ensemble      import AdaBoostRegressor
from sklearn.ensemble      import GradientBoostingRegressor
from xgboost               import XGBRegressor
from lightgbm              import LGBMRegressor
from catboost              import CatBoostRegressor



def tree_classifiers():
    tree_classifiers = {
    "Decision Tree": DecisionTreeRegressor(),
    "Extra Trees":   ExtraTreesRegressor(n_estimators=100),
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "AdaBoost":      AdaBoostRegressor(n_estimators=100),
    "Skl GBM":       GradientBoostingRegressor(n_estimators=100),
    "XGBoost":       XGBRegressor(n_estimators=100),
    "LightGBM":      LGBMRegressor(n_estimators=100),
    "CatBoost":      CatBoostRegressor(n_estimators=100),
    }

    cat_vars = ['season','is_weekend','is_holiday','year','month','weather_code']
    num_vars = ['t1','t2','hum','wind_speed']
    num_prepro = pipeline.Pipeline(steps=[('imputer', SimpleImputer(strategy='constant',fill_value=-9999))])
    cat = pipeline.Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value=-9999)),('ordinal', OrdinalEncoder())])
    tree_pre  = ColumnTransformer(transformers=[('num', num_prepro, num_vars),('cat', cat, cat_vars)], remainder='drop')

    tree_classifiers = {name: pipeline.make_pipeline(tree_pre, model) for name, model in tree_classifiers.items()}
    return tree_classifiers


def enhance(df):
    dfc = df.copy()

    for season in dfc['season'].unique():
        seasonal_data = dfc[dfc['season'] == season]
        hum_std = seasonal_data['hum'].std()
        wind_speed_std = seasonal_data['wind_speed'].std()
        t1_std = seasonal_data['t1'].std()
        t2_std = seasonal_data['t2'].std()

        for i in dfc[dfc['season'] == season].index:
            if np.random.randint(2) == 1:
                dfc['hum'].values[i] +=hum_std/50
            else:
                dfc['hum'].values[i] -= hum_std/50

            if np.random.randint(2) == 1:
                dfc['wind_speed'].values[i] += wind_speed_std/50
            else:
                dfc['wind_speed'].values[i] -= wind_speed_std/50

            if np.random.randint(2) == 1:
                dfc['t1'].values[i] += t1_std/50
            else:
                dfc['t1'].values[i] += t1_std/50

            if np.random.randint(2) == 1:
                dfc['t2'].values[i] += t2_std/50
            else:
                dfc['t2'].values[i] += t2_std/50
            
    return dfc