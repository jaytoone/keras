import numpy as np
import pandas as pd


x = np.arange(1, 101)
y = np.arange(1, 101)

data_len = len(x)
train_len = int(data_len * 0.6)
val_len = int(data_len * 0.2)
# test_len = int(data_len * 0.2)

x_train = x[:train_len]
y_train = y[:train_len]

x_val = x[train_len:train_len + val_len]
y_val = y[train_len:train_len + val_len]

x_test = x[train_len + val_len:]
y_test = y[train_len + val_len:]

print(x_train.shape)
print(x_val.shape)
print(x_test.shape)
# quit()

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

# model.add(Dense(32, input_dim=1))
model.add(Dense(32, input_shape=(1, )))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=50, batch_size=1, validation_data=(x_val, y_val))
# model.fit(x, y, epochs=100)

loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print('mse :', mse)

pred_x = np.arange(120, 123, 1)
print(model.predict(pred_x))

model.summary()