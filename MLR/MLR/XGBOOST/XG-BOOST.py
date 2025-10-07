import pandas as pd #for reading dataset
import numpy as np # array handling functions
import xgboost as xgb
from sklearn.metrics import accuracy_score


dataset = pd.read_csv("Rainfall.csv")#reading dataset
#print(dataset) # printing dataset

x = dataset.iloc[:,:-1].values #locating inputs
y = dataset.iloc[:,-1].values #locating outputs

#printing X and Y
print("x=",x)
print("y=",y)

from sklearn.model_selection import train_test_split # for splitting dataset
x_train,x_test,y_train,y_test = train_test_split(x ,y, test_size = 0.25 ,random_state = 0)
#printing the spliited dataset
print("x_train=",x_train)
print("x_test=",x_test)
print("y_train=",y_train)
print("y_test=",y_test)

#importing algorithm
model = xgb.XGBRegressor()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(y_pred)




a1=float(input("ENTER THE Day= "))
b1=float(input("ENTER THE Pressure= "))
c1=float(input("ENTER THE Maxtemp= "))
d1=float(input("ENTER THE Temp= "))
e1=float(input("ENTER THE Mintemp= "))
f1=float(input("ENTER THE dewpoint= "))
g1=float(input("ENTER THE Humidity= "))
h1=float(input("ENTER THE Cloud= "))
i1=float(input("ENTER THE Sunshine= "))
j1=float(input("ENTER THE Wind Direction= "))
k1=float(input("ENTER THE Wind Speed= "))



a = model.predict([[a1,b1,c1,d1,e1,f1,g1,h1,i1]])
print('Predicted Rainfall Today: %s' % int(a))

if a==0:
    print("No")
else:
    print("Yes")



