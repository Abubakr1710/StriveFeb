import pandas as pd
import numpy as np
import split as sp
from sklearn import metrics
import time


data =pd.read_csv('Chapter 02/15. TimeSeries/climate.csv')
data =data.drop(['Date Time'], axis=1)
#print(data)

def paring(data, seq_len=5):
    x=[]
    y=[]
    for i in range(0, (data.shape[0]-(seq_len+1)), seq_len+1):
        seq= np.zeros((seq_len, data.shape[1]))
        for j in range(seq_len):
            seq[j]=data.values[i+j]

        x.append(seq.flatten())
        y.append(data["T (degC)"][i+seq_len])
    
    return np.array(x), np.array(y)

x, y = paring(data)
# print(data.shape)
print(x.shape)
# print(y[0])
# print(y[1])
# print(len(y))
# print(len(data))
# print(len(y)/len(data))
print(x.shape)


# X_train, X_test, y_train, y_test = sp.splitting(x,y)

# results = pd.DataFrame({'Model': [], 'MSE': [], 'MAB': [], " % error": [], 'Time': [],})
# tree_classifiers = sp.tree_regressors()

# for model_name, model in tree_classifiers.items():
#     rang = abs(y_train.max()) + abs(y_train.min())

#     start_time = time.time()
#     model.fit(X_train, y_train.ravel())
#     total_time = time.time() - start_time
        
#     pred = model.predict(X_test)
    
#     results = results.append({"Model":    model_name,
#                               "MSE": metrics.mean_squared_error(y_test, pred),
#                               "MAB": metrics.mean_absolute_error(y_test, pred),
#                               " % error": metrics.mean_squared_error(y_test, pred) / rang,
#                               "Time":     total_time
#                               },
#                               ignore_index=True)



# results_ord = results.sort_values(by=['MSE'], ascending=True, ignore_index=True)
# results_ord.index += 1 
# results_ord.style.bar(subset=['MSE', 'MAE'], vmin=0, vmax=100, color='#5fba7d')

#print(results_ord)

