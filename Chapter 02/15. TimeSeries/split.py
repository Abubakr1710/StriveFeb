from sklearn.model_selection import train_test_split


from sklearn.tree          import DecisionTreeRegressor
from sklearn.ensemble      import RandomForestRegressor
from sklearn.ensemble      import ExtraTreesRegressor
from sklearn.ensemble      import AdaBoostRegressor
from sklearn.ensemble      import GradientBoostingRegressor
from xgboost               import XGBRegressor
from lightgbm              import LGBMRegressor
from catboost              import CatBoostRegressor
from sklearn.linear_model  import LinearRegression


def splitting(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    return X_train, X_test ,y_train, y_test



def tree_regressors():
    tree_regressor = {
    'Linear':        LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Extra Trees":   ExtraTreesRegressor(n_estimators=100),
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "AdaBoost":      AdaBoostRegressor(n_estimators=100),
    "Skl GBM":       GradientBoostingRegressor(n_estimators=100),
    "XGBoost":       XGBRegressor(n_estimators=100),
    "LightGBM":      LGBMRegressor(n_estimators=100),
    "CatBoost":      CatBoostRegressor(n_estimators=100),
    }
    return tree_regressor