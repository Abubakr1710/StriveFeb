
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from train import mod
while True:

    age = int(input("How old are you? \n"))
    sex = input('what is your sex? \n')
    child = int(input("How many children do you have? \n"))
    smoker = bool(input("Do you smoke? \n"))
    bmi = float(input("what is your body bmi index? \n"))
    region = input('which region do you live in? (southwest,southeast,northwest,northeast) \n')

    ndata = pd.DataFrame({
        'age':[age],
        'sex':[sex],
        'bmi':[bmi],
        'children':[child],
        'smoker':[smoker],
        'region': [region]
    })
    
    X = ndata.values
    encoder = ColumnTransformer( [('ordinal', OrdinalEncoder(handle_unknown= 'use_encoded_value', unknown_value = -1), [1,4,5] )] )
    X = np.concatenate((X[:,[0,2,3]],encoder.fit_transform(X)),axis=1)


    pred = mod.predict(X)
    print(*pred.round())
    cont = str(input('Do you want to recalculate?(yes/no) \n'))
    if cont == 'no':
        break
    else: 
        continue
    
    '''
    Preprocess
    predict
    
    '''
    #print("You are too fucked up 1 milly")