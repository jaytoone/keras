import numpy as np
from keras.models import Sequential  
from keras.layers import Dense, LSTM

x = np.array([[1, 2, 3], [2, 3, 4], [4, 5, 6], [5, 6, 7], [6, 7, 8]])
y = np.arange(4, 9, 1)

x = x.reshape(x.shape[0], x.shape[1], 1)

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3, 1)))
model.add(Dense(5))
model.add(Dense(1))

model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x, y, epochs=100, batch_size=1)
loss, mse = model.evaluate(x, y, batch_size=1)

print(loss, mse)

x_input = np.array([6, 7, 8])