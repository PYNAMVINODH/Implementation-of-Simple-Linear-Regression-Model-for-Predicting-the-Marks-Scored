# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: PYNAM VINODH
RegisterNumber:  212223240131
import pandas as pd
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("student_scores.csv")
#displaying the content 
print(df.head())
reg=linear_model.LinearRegression()
x=df.iloc[:,:-1].values
y=df.iloc[:,1].values
print(x)
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
*/
```
## Output:
![image](https://github.com/PYNAMVINODH/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742678/f8bbca6b-f95b-45f0-8725-c2fe1ea5ac5c)


```
plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
## Output:
![image](https://github.com/PYNAMVINODH/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742678/34de95d4-86e0-4a55-92be-f4d962797636)




```
plt.scatter(x_test,y_test,color="red")
plt.plot(x_test,regressor.predict(x_test),color="blue")
plt.title("Hours vs Scores (testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```
## Output:
![image](https://github.com/PYNAMVINODH/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742678/b9f8eaba-35e7-4a09-8b78-e7ee98d2ec49)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
