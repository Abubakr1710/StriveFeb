from pyexpat import model
from sklearn.impute import  SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn import pipeline
from sklearn import compose



from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.ensemble      import ExtraTreesClassifier
from sklearn.ensemble      import AdaBoostClassifier
from sklearn.ensemble      import GradientBoostingClassifier
from sklearn.experimental  import enable_hist_gradient_boosting 
from sklearn.ensemble      import HistGradientBoostingClassifier
from xgboost               import XGBClassifier
from lightgbm              import LGBMClassifier
from catboost              import CatBoostClassifier



def tree_classifiers():
    tree_classifiers = {
    "Decision Tree": DecisionTreeClassifier(),
    "Extra Trees": ExtraTreesClassifier(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Skl GBM": GradientBoostingClassifier(),
    "Skl HistGBM":HistGradientBoostingClassifier(),
    "XGBoost": XGBClassifier(),
    "LightGBM": LGBMClassifier(),
    "CatBoost": CatBoostClassifier()}

    cat_vars  = ['Sex', 'Embarked', 'Title']
    num_vars  = ['Pclass', 'SibSp', 'Parch', 'Fare', 'Age']
    num_prepro = pipeline.Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent'))])
    cat = pipeline.Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('ohe', OneHotEncoder(handle_unknown='ignore'))])
    tree_pre  = compose.ColumnTransformer(transformers=[('num', num_prepro, num_vars),('cat', cat, cat_vars)], remainder='drop')

    tree_classifiers = {name: pipeline.make_pipeline(tree_pre, model) for name, model in tree_classifiers.items()}
    return tree_classifiers



