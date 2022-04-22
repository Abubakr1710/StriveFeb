from multiprocessing.spawn import import_main_path
import pandas as pd
import numpy as np
from torch import int64
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn import pipeline
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.metrics import accuracy_score
import time
import data_handler as dh

np.random.seed(0)

data = pd.read_csv('Chapter 02/11. Data Enhancement/london_merged.csv')
#print(data)

data['year'] = data['timestamp'].apply(lambda row: row[:4])
data['year'] = data['year'].astype(int)
data['month'] = data['timestamp'].apply(lambda row: row.split('-')[2][:2] )
data['month'] = data['month'].astype(int)
data['hour'] = data['timestamp'].apply(lambda row: row.split(':')[0][-2:] )
data['hour'] = data['hour'].astype(int)
data.drop('timestamp', axis=1, inplace=True)
#print(data)

x,y = data.drop(['cnt'], axis=1), data['cnt']

enh = dh.enhance(data)

x_train, x_val, y_train, y_val = train_test_split(x, y,
                                    test_size=0.2,
                                    random_state=0  # Recommended for reproducibility
                                )
x_train_old_len = x_train

ext_sam = enh.sample(enh.shape[0] // 4)
x_train = pd.concat([x_train, ext_sam.drop(['cnt'], axis=1 ) ])
y_train = pd.concat([y_train, ext_sam['cnt'] ])



transformer = PowerTransformer()
y_train = transformer.fit_transform(y_train.values.reshape(-1,1))
y_val = transformer.transform(y_val.values.reshape(-1,1))

rang = abs(y_train.max()) + abs(y_train.min())

tree_classifiers = dh.tree_classifiers()

results = pd.DataFrame({'Model': [], 'MSE': [], 'MAB': [], " % error": [], 'Time': [],})

for model_name, model in tree_classifiers.items():
    
    start_time = time.time()
    model.fit(x_train, y_train.ravel())
    total_time = time.time() - start_time
        
    pred = model.predict(x_val)
    
    results = results.append({"Model":    model_name,
                              "MSE": metrics.mean_squared_error(y_val, pred),
                              "MAB": metrics.mean_absolute_error(y_val, pred),
                              " % error": metrics.mean_squared_error(y_val, pred) / rang,
                              "Time":     total_time
                              },
                              ignore_index=True)
### END SOLUTION


results_ord = results.sort_values(by=['MSE'], ascending=True, ignore_index=True)
results_ord.index += 1 
results_ord.style.bar(subset=['MSE', 'MAE'], vmin=0, vmax=100, color='#5fba7d')

print(results_ord)

print('ytrain max:',y_train.max())
print('ytrain min:',y_train.min())
print(y_val[5])
print(tree_classifiers['Random Forest'].predict(x_val)[5])
print('Train part size is increased by:',round(len(x_train)/len(x_train_old_len),1),'times after Data Enhancement')