import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle,joblib
from joblib import dump, load
#from sklearn.externals import joblib
data=pd.read_csv('C:/Users/prana/Desktop/final year project/Fertilizer Prediction.csv')
data_X=data.iloc[:,:-1]
data_y=data.iloc[:,-1]

label= pd.get_dummies(data.FertilizerName).iloc[: , 1:]
data= pd.concat([data,label],axis=1)


data.drop('FertilizerName', axis=1,inplace=True)
data.drop('SoilType', axis=1,inplace=True)
data.drop('CropType', axis=1,inplace=True)
#pd.set_option('display.max_columns', None)
#print(data.head())
train=data.iloc[:, 0:6].values
test=data.iloc[: ,6:].values
X_train,X_test,y_train,y_test=train_test_split(train,test,test_size=0.3,random_state=1)

X_train=list(X_train)
for i in range(len(X_train)):
    X_train[i]=[int(j) for j in X_train[i]]
    
X_test=list(X_test)
for i in range(len(X_test)):
    X_test[i]=[int(j) for j in X_test[i]]

y_train=list(y_train)
for i in range(len(y_train)):
    y_train[i]=[int(j) for j in y_train[i]]
    
y_test=list(y_test)
for i in range(len(y_test)):
    y_test[i]=[int(j) for j in y_test[i]]


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
algo=RandomForestClassifier(n_estimators=10,random_state=10)

algo.fit(X_train,y_train)

pred1=algo.predict(X_test)
print(pred1)
print("accuracy is",algo.score(X_test,y_test)*100,'%')



"""from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)"""

with open('mysaved_md_pickle','wb') as file:
    pickle.dump(algo,file)
joblib.dump(algo, 'mysaved_md_pickle.pkl')

with open('mysaved_md_pickle','rb') as file:
    loaded_model=pickle.load(file)
clf = joblib.load('mysaved_md_pickle.pkl')
print(loaded_model.predict([[32,55,40,22,12,20]]))