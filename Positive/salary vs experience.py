# #importing libraries
import numpy as np #numerical python
import pandas as pd #to open CSV file
import matplotlib.pyplot as plt #graphical representation


dataset = pd.read_csv('AbsvsGrade.csv')
print(dataset)
X = dataset.iloc[:,:-1].values
print(X)
y = dataset.iloc[:,-1].values
print(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .20, random_state = 0)##from sklearn.linear_model import LinearRegression
##from sklearn.model_selection import LinearRegression 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)#y=c+bx
y_pred = regressor.predict(X_test)
print(y_pred)    
##Visualising the training set result
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Abs vs Grade (Training set)')
plt.xlabel('Abs')
plt.ylabel('Grade')
plt.show()
##
plt.scatter(X_test, y_test, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Abs vs Grade(Test Set)')
plt.xlabel('Abs')
plt.ylabel('Grade')
plt.show()
##
