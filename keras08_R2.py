
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense


x = np.arange(1, 101)
y = np.arange(1, 101)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=0)

print(len(x_train) / len(x))
print(len(x_val) / len(x))
print(len(x_test) / len(x))

loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print('mse :', mse) 

x_pred = np.arange(100, 105, 1)
result = model.predict(x_pred, batch_size=1)
print(result)

from sklearn.metrics import mean_squared_error, r2_score

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE :', RMSE(y_test, y_pred))

R2_score = r2_score(y_test, y_pred)
print('R2_score :', R2_score)