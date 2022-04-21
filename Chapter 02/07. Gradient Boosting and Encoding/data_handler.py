from tkinter import Y
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

hello = "<3"

def get_data(pth):

    data = pd.read_csv(pth)

    x_train, x_test, y_train, y_test = train_test_split(data.values[:,:-1], data.values[:,-1], test_size=0.3, random_state = 42, stratify=data.values[:,4])

    ct = ColumnTransformer( [('ordinal', OrdinalEncoder(handle_unknown= 'use_encoded_value', unknown_value = -1), [1,4,5] )] )
    x_train = np.concatenate((x_train[:,[0,2,3]],ct.fit_transform(x_train)),axis=1)
    x_test = np.concatenate((x_test[:,[0,2,3]],ct.transform(x_test)),axis=1)


    return x_train, x_test, y_train, y_test