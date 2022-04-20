import data_handler as dh
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score

rf = RandomForestRegressor()
ada = AdaBoostRegressor()
Gr = GradientBoostingRegressor()
xgb = XGBRegressor()

x_train, x_test, y_train, y_test = dh.get_data("Chapter 02/07. Gradient Boosting and Encoding/insurance.csv")

#print(dh.hello)

def model(x_train, y_train, x_test, y_test):
    model_sel = [rf, ada, Gr, xgb]
    score = []
    for mod_name in model_sel:
        clf = mod_name
        clf.fit(x_train, y_train)
        acc = clf.score(x_test, y_test)
        score.append(acc)
    return score
    
sc = model(x_train, y_train, x_test, y_test)
print(sc)