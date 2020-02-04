import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

x = [np.arange(i, i+3, 1) for i in range(13)]
x = np.array(x)
print(x.shape)

y = np.array(np.arange(13))
print(y.shape)
x_ = x
x = x.reshape(-1, 3, 1)
# y = y.reshape(-1, 1, 1)

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3, 1), return_sequences=True))
model.add(LSTM(10, activation='relu'))
model.add(Dense(5))
model.add(Dense(1))

input = Input(shape=(3,))
model2 = Dense(10)(input)
model2 = Dense(5)(model2)
output = Dense(1)(model2)
model2 = Model(inputs=input, outputs=output)


model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x, y, epochs=100, batch_size=1)
loss, mse = model.evaluate(x, y, batch_size=1)

from keras.callbacks import EarlyStopping

callback = EarlyStopping(monitor='loss', patience=20, mode='auto')
# callback = EarlyStopping(monitor='acc', patience=20, mode='max')
model2.compile(loss='mse', optimizer='adam', metrics=['mse'])
model2.fit(x_, y, epochs=1000, batch_size=1, callbacks=[callback])
loss2, mse2 = model2.evaluate(x_, y, batch_size=1, verbose=1)

x_input = np.array([[6.5, 7.5, 8.5], [50, 60, 70], [70, 80, 90], [100, 110, 120]])
x = x_input.reshape(-1, 3, 1)

y_pred = model.predict(x)
print(loss, mse)
print(loss2, mse2)
print(y_pred)
