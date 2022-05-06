# Importng needed libraries
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble      import GradientBoostingRegressor
from xgboost               import XGBRegressor
from lightgbm              import LGBMRegressor
from sklearn.linear_model  import LinearRegression


from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from joblib import Memory
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

#print(x.shape)
#print(y.shape)
#----------------------------------------------------------------------------------#

#Feature extraction
def get_features(x):
    feature = []
    for i in range(x.shape[0]):
        mean_column_1 = np.mean(x[i, :, 0])
        std_column_2 =np.std(x[i, :, 1])
        mean_column_3 =np.mean(x[i, :, 2])
        std_column_4 =np.std(x[i, :, 3])
        median_column_5 =np.median(x[i, :, 4])
        median_column_6 =np.median(x[i, :, 5])
        median_column_7 =np.median(x[i, :, 6])
        max_column_8 =np.max(x[i, :, 7])
        mean_column_9 =np.mean(x[i, :, 8])
        min_column_10=np.min(x[i, :, 9])
        median_column_11=np.median(x[i, :, 10])
        std_column_12=np.std(x[i, :, 11])
        mean_column_13=np.mean(x[i, :, 12])
        mean_column_14=np.mean(x[i, :, 13])

        feature.append((mean_column_1, std_column_2,mean_column_3,std_column_4,median_column_5,median_column_6,median_column_7,
                        max_column_8,mean_column_9,min_column_10,median_column_11,std_column_12,mean_column_13,mean_column_14))
        #feature =np.hstack((mean_column_1, std_columns_2))
    return np.array(feature)

nx = get_features(x)
#print(nx.shape)
#----------------------------------------------------------------------------------------------------------------------------------------#

# Splitting the data
def split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test =split(nx, y)

scaler =StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
mod =LGBMRegressor()
mod.fit(X_train, y_train)
pred = mod.predict(X_test)
print(metrics.mean_squared_error(y_test, pred))
