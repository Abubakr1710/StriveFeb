import pandas as pd
import numpy as np
from train import best_model
model  = best_model
while True:

    age = int(input("How old are you? \n"))
    sex = str(input('what is your sex? \n'))
    child = int(input("How many children do you have? \n"))
    smoker = bool(input("Do you smoke? \n"))
    bmi = float(input("what is your body bmi index? \n"))
    region = str(input('which region do you live in? (southwest,southeast,northwest,northeast) \n'))

    ndata = pd.DataFrame({
        'age':[age],
        'sex':[sex],
        'bmi':[bmi],
        'children':[child],
        'smoker':[smoker],
        'region': [region]
    })
    
    gb = model[0]
    tr = model[1]

    data_transform = tr.transform(ndata.values)
    pred = np.array(gb.predict(data_transform))
    print(pred)


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