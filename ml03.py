import pandas as pd
import numpy as np


csv_data = pd.read_csv('data\pima-indians-diabetes.csv', delimiter=',')
print(type(csv_data))
csv_data = np.loadtxt('data\pima-indians-diabetes.csv', delimiter=',')
print(type(csv_data))

x = csv_data[:, 0:8]
y = csv_data[:, 8]

from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(x, y, epochs=100, batch_size=10)

print('ACC : %.4f' % (model.evaluate(x, y)[1]))