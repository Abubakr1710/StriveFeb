import numpy as np
import pandas as pd

df = pd.read_csv('climate.csv')
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

def get_features(x):
    feature = []
    for i in range(x.shape[0]):
        mean_column_1 = np.mean(x[i, :, 0])
        std_columns_2 =np.std(x[i, :, 1])
        feature.append((mean_column_1, std_columns_2))
        #feature =np.hstack((mean_column_1, std_columns_2))
    return np.array(feature)

nx = get_features(x)
print(nx.shape)


    







