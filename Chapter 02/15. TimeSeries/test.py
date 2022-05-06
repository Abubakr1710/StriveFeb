from matplotlib.pyplot import axes, axis
import numpy as np
import pandas as pd

df = pd.read_csv('Chapter 02/15. TimeSeries/climate.csv')
df = df.drop(['Date Time'], axis=1)

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

def get_feat(x):
    nx =[]
    for i in range(x.shape[0]):
        meanx = x[i].mean(axis=0)
        nx.append(meanx)
        #gr =[]

        #for j in range(x.shape[2]):
        #    gr.append(np.mean(x[i][:,j]))
    
        # nx.append(gr)
    return np.array(nx)
nx =get_feat(x)
print(nx.shape)