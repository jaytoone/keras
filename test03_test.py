import numpy as np
import pandas as pd


x_train = np.arange(10)
y_train = np.arange(10)

x_test = np.arange(10, 21, 1)
y_test = np.arange(10, 21, 1)

print(x_test, y_test)

# print(x.shape)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

# model.add(Dense(32, input_dim=1))
model.add(Dense(32, input_shape=(1, )))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=50, batch_size=1)
# model.fit(x, y, epochs=100)

loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print('mse :', mse)

pred_x = np.arange(11, 15, 1)
print(model.predict(pred_x))

model.summary()