import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data =pd.read_csv('Chapter 02/15. TimeSeries/climate.csv')
data =data.drop(['Date Time'], axis=1)
#print(data)

def paring(data, seq_len=6):
    x=[]
    y=[]
    for i in range(0, (data.shape[0]-(seq_len+1)), seq_len+1):
        seq= np.zeros((seq_len, data.shape[1]))
        for j in range(seq_len):
            seq[j]=data.values[i+j]

        x.append(seq.flatten())
        y.append(data["T (degC)"][i+seq_len])
    
    return np.array(x), np.array(y)


print(data.shape)
x, y = paring(data)
print(x.shape)
print(y[0])
print(y[1])

# train_test_split