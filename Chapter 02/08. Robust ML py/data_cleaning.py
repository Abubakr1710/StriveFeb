import pandas as pd

def cleaned_data():
    df      = pd.read_csv("Chapter 02/08. Robust ML py/train.csv", index_col='PassengerId')
    df_test = pd.read_csv("Chapter 02/08. Robust ML py/test.csv",  index_col='PassengerId')

    #Extract the title
    get_Title_from_Name = lambda x : x.split(',')[1].split('.')[0].strip()
    df['Title'] = df['Name'].apply(get_Title_from_Name)
    df_test['Title'] = df_test['Name'].apply(get_Title_from_Name)

    title_dictionary = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir" : "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess":"Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr" : "Mr",
        "Mrs" : "Mrs",
        "Miss" : "Miss",
        "Master" : "Master",
        "Lady" : "Royalty"
    }

    df["Title"] = df['Title'].map(title_dictionary)
    df_test["Title"] = df_test["Title"].map(title_dictionary)
    
    x = df.drop(columns=["Survived", 'Name', 'Ticket', 'Cabin'])
    y = df["Survived"]
    x_test = df_test.drop(columns=['Name', 'Ticket', 'Cabin'])
    return x,y,x_test