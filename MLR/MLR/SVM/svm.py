import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

dataset = pd.read_csv("air pollution.csv")
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
a1=float(input("ENTER THE Temp= "))
b1=float(input("ENTER THE Humidity= "))
c1=float(input("ENTER THE PM2.5= "))
d1=float(input("ENTER THE PM10= "))
e1=float(input("ENTER THE NO2= "))
f1=float(input("ENTER THE SO2= "))
g1=float(input("ENTER THE CO= "))
h1=float(input("ENTER THE Proximity= "))
i1=float(input("ENTER THE Population= "))


a=  regressor.predict([[a1,b1,c1,d1,e1,f1,g1,h1,i1]])
print('Predicted new output value: %s' % (a))

if a==0:
    print("Poor")
elif a==1:
    print("Good")

elif a==2:
    print("Hazadrous")

elif a==3:
    print("Moderate")






















