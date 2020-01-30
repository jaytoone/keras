from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense


x = np.array([np.arange(1, 101), np.arange(101, 201)])
y = np.array([np.arange(1, 101), np.arange(101, 201)])

x = x.reshape(-1, 2)
y = y.reshape(-1, 2)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=0)

print(len(x_train) / len(x))
print(len(x_val) / len(x))
print(len(x_test) / len(x))

model = Sequential()
model.add(Dense(32, input_shape=(2, )))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(2))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=50, batch_size=1, validation_data=(x_val, y_val))

loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print('mse :', mse) 

x_pred = np.arange(100, 110, 1).reshape(-1, 2)
result = model.predict(x_pred, batch_size=1)
print(result)