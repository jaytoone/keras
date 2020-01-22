import numpy as np
import pandas as pd

x = np.arange(10)
y = np.arange(10)

# print(x.shape)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(32, input_dim=1))
model.add(Dense(32))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x, y, epochs=1000, batch_size=1)
# model.fit(x, y, epochs=100)

loss, mse = model.evaluate(x, y, batch_size=1)
print('mse :', mse)

pred_x = np.arange(10, 20, 1)
print(model.predict(pred_x))