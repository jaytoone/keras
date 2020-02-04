from keras22_univeriate2 import split_sequence
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

input_seq1 = np.array(np.arange(10, 101, 10))
input_seq2 = np.array(np.arange(15, 106, 10))

output_seq = np.array([input_seq1[i] + input_seq2[i] for i in range(len(input_seq1))])
print(output_seq)

input_seq1 = input_seq1.reshape(len(input_seq1), 1)
input_seq2 = input_seq2.reshape(len(input_seq2), 1)
output_seq = output_seq.reshape(len(output_seq), 1)

dataset = np.hstack((input_seq1, input_seq2, output_seq))
x, y = split_sequence(dataset, 3)

print(x.shape, y.shape)

x = x.reshape(-1, 9, 1)

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(9, 1), return_sequences=False))
model.add(Dense(5))
model.add(Dense(3))

model.summary()

from keras.callbacks import EarlyStopping


callback = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x, y, epochs=100, batch_size=1, callbacks=[callback])
loss, mse = model.evaluate(x, y, batch_size=1)

print(loss, mse)

x_pred = np.array([90, 100, 110])
x_pred = np.vstack((x_pred, x_pred, x_pred))
x_pred = x_pred.reshape(-1, 9, 1)
y_pred = model.predict(x_pred)

print(y_pred)


# 10042.715994698661 10042.715994698661
# [[5.049321 6.74885  7.699828]]

# 139.77554150990076 139.77554150990076
# [[106.8416   113.977974 220.85042 ]]