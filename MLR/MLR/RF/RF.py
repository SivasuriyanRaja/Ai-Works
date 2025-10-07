import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

dataset = pd.read_csv("Rainfall.csv")
print(dataset)


x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

print(x)
print(y)


from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x ,y, test_size = 0.30 ,random_state = 0)
print("x_train=",x_train)
print("x_test=",x_test)
print("y_train=",y_train)
print("y_test=",y_test)


# SUPPORT VECTOR MACHINE REGRESSION
from sklearn.svm import SVR
regressor = SVR()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
print(y_pred)

svm = mean_absolute_error(y_test, y_pred)
print("support vector machine learning accuracy level is = ",svm/10)



from sklearn.ensemble import RandomForestRegressor  
Regressor= RandomForestRegressor(n_estimators = 20)  
Regressor.fit(x_train, y_train)
y_pred1=Regressor.predict(x_test) #testing model
print("y_pred",y_pred1) 

svm1 = mean_absolute_error(y_test, y_pred1)
print("Random Forest Regressor learning accuracy level is = ",svm1/10)

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

a=Regressor.predict([[a1,b1,c1,d1,e1,f1,g1,h1,i1,j1,k1]])
print('Predicted Rainfall Today: %s' % int(a))

if a==0:
    print("No")
elif a==1:
    print("Yes")
else:
    print("Invalid")




















