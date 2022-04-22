from unittest import result
import pandas as pd
import data_handler as dh
import data_cleaning as dc
from sklearn.model_selection import train_test_split
import time
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, cross_val_predict

x,y,x_test = dc.cleaned_data()
tree_classifiers =dh.tree_classifiers()

def model_results():

    X_train, X_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=0, stratify=y)

    results = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})

    for model_name, model in tree_classifiers.items():
        start_time = time.time()        
        model.fit(X_train,y_train)
        pred =model.predict(X_val)

        total_time = time.time() - start_time

        results = results.append({"Model":    model_name,
                                "Accuracy": metrics.accuracy_score(y_val, pred)*100,
                                "Bal Acc.": metrics.balanced_accuracy_score(y_val, pred)*100,
                                "Time":     total_time},
                                ignore_index=True)
                                
    results_ord = results.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
    results_ord.index += 1 
    results_ord.style.bar(subset=['Accuracy', 'Bal Acc.'], vmin=0, vmax=100, color='#5fba7d')
    return results_ord
# mod_res = model_results()
# print(mod_res)


def mod_res_kfold():
    skf =StratifiedKFold(n_splits=10,shuffle=True, random_state=0,)

    results = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})

    for model_name, model in tree_classifiers.items():
        start_time = time.time()
            
        # TRAIN AND GET PREDICTIONS USING cross_val_predict() and x,y
        pred = cross_val_predict(model, x, y, cv=skf)

        total_time = time.time() - start_time

        results = results.append({"Model":    model_name,
                                "Accuracy": metrics.accuracy_score(y, pred)*100,
                                "Bal Acc.": metrics.balanced_accuracy_score(y, pred)*100,
                                "Time":     total_time},
                                ignore_index=True)
                                
                                



    results_ord = results.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
    results_ord.index += 1 
    results_ord.style.bar(subset=['Accuracy', 'Bal Acc.'], vmin=0, vmax=100, color='#5fba7d')
    return results_ord

# res_kfold = mod_res_kfold()
# print(res_kfold)

def mod():
    best_model = tree_classifiers['Skl GBM']
    best_model.fit(x,y)
    test_pred = best_model.predict(x_test)
    sub1 = pd.DataFrame({
        'Survived': test_pred,
        'sex': x_test['Sex']
        })
    return sub1
ans = mod()
print(ans)