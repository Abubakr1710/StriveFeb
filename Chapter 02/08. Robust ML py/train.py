from turtle import pd
import pandas as pd
import testhandler as dh
import time
from sklearn import metrics

#Loading data
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

x = df.drop(columns=["Survived", 'Name', 'Ticket', 'Cabin']) # X DATA (WILL BE TRAIN+VALID DATA)
y = df["Survived"] # 0 = No, 1 = Yes

x_test = df_test.drop(columns=['Name', 'Ticket', 'Cabin']) # # X_TEST DATA (NEW DATA)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=0, stratify=y)


results = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})
tree_classifier =dh.tree_classifiers()

for model_name, model in tree_classifier.items():
    start_time = time.time()
    
    # FOR EVERY PIPELINE (PREPRO + MODEL) -> TRAIN WITH TRAIN DATA (x_train)
    
    model.fit(X_train,y_train)
    # GET PREDICTIONS USING x_val
    pred =model.predict(X_val)

    total_time = time.time() - start_time

    results = results.append({"Model":    model_name,
                              "Accuracy": metrics.accuracy_score(y_val, pred)*100,
                              "Bal Acc.": metrics.balanced_accuracy_score(y_val, pred)*100,
                              "Time":     total_time},
                              ignore_index=True)
                              
results_ord = results.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
results_ord.index += 1 
print(results_ord.style.bar(subset=['Accuracy', 'Bal Acc.'], vmin=0, vmax=100, color='#5fba7d'))