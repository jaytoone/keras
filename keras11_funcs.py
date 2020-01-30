from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Input


x = np.array([np.arange(1, 101), np.arange(101, 201), np.arange(301, 401)])
y = np.array([np.arange(1, 101)])

x = x.reshape(-1, 3)
y = y.reshape(-1, 1)
print(x.shape, y.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=0)

print(len(x_train) / len(x))
print(len(x_val) / len(x))
print(len(x_test) / len(x))

# model = Sequential()
# model.add(Dense(32, input_shape=(3, )))
# model.add(Dense(128))
# model.add(Dense(32))
# model.add(Dense(1))

#   Functional API
# input = Input(shape=(3,))
# model = Dense(32)(input)
# model = Dense(32)(model)
# model = Dense(32)(model)
# output = Dense(1)(model)
# model = Model(inputs=input, outputs=output)

#   Functional API
input = Input(shape=(3,))
model = Dense(32)(input)
model = Dense(32)(model)
model = Dense(32)(model)
output = Dense(1)(model)
model = Model(inputs=input, outputs=output)

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))

loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print('mse :', mse) 

y_pred = model.predict(x_test, batch_size=1)
print(y_test.shape, y_pred.shape)
# quit()

from sklearn.metrics import mean_squared_error, r2_score

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE :', RMSE(y_test, y_pred))

R2_score = r2_score(y_test, y_pred)
print('R2_score :', R2_score)