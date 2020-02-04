from keras22_univeriate2 import split_sequence
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

dataset = np.arange(10, 101, 10)
print(dataset)

x, y = split_sequence(dataset, 3)
x = x.reshape(-1, 3, 1)
print(x.shape, y.shape)



model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3, 1), return_sequences=False))
model.add(Dense(5))
model.add(Dense(1))

model.summary()

from keras.callbacks import EarlyStopping


callback = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x, y, epochs=100, batch_size=1, callbacks=[callback])
loss, mse = model.evaluate(x, y, batch_size=1)

print(loss, mse)

x_pred = np.array([90, 100, 110]).reshape(-1, 3, 1)
y_pred = model.predict(x_pred)

print(y_pred)