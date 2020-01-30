import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

x = [np.arange(i, i+3, 1) for i in range(13)]
x = np.array(x)
print(x.shape)

x2 = [np.arange(10 * i, 10 * i + 10 * 2 + 1, 10) for i in range(13)]
x2 = np.array(x2)
print(x2.shape)

y = np.array(np.arange(13))
y2 = np.array(np.arange(10, 140, 10))
print(y2.shape)
# quit()

x_ = x
x = x.reshape(-1, 3, 1)
x2 = x2.reshape(-1, 3, 1)
# y = y.reshape(-1, 1, 1)

# model = Sequential()
# model.add(LSTM(10, activation='relu', input_shape=(3, 1)))
# model.add(Dense(5))
# model.add(Dense(1))

input = Input(shape=(3, 1))
# model2 = Dense(10)(input)
model = LSTM(10, activation='relu')(input)
model = Dense(5)(model)
# output = Dense(1)(model)
# model2 = Model(inputs=input, outputs=output)

input2 = Input(shape=(3, 1))
model2 = LSTM(10, activation='relu')(input)
model2 = Dense(5)(model2)
# output2 = Dense(1)(model2)

from keras.layers.merge import concatenate

# merge = concatenate([model, model2])
merge = Add([model, model])

merge = Dense(8)(merge)
output = Dense(1)(merge)

output_1 = Dense(4)(output)
output_1 = Dense(1)(output_1) 

output_2 = Dense(4)(output)
output_2 = Dense(1)(output_2) 

model = Model(inputs=[input, input2], outputs=[output_1, output_2])
model.summary()
# quit()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x, x2], [y, y2], epochs=100, batch_size=1)
res = model.evaluate([x, x2], [y, y2], batch_size=1)

from keras.callbacks import EarlyStopping

# callback = EarlyStopping(monitor='loss', patience=20, mode='auto')
# # callback = EarlyStopping(monitor='acc', patience=20, mode='max')
# model2.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model2.fit(x_, y, epochs=1000, batch_size=1, callbacks=[callback])
# loss2, mse2 = model2.evaluate(x_, y, batch_size=1, verbose=1)

# x_input = np.array([[6.5, 7.5, 8.5], [50, 60, 70], [70, 80, 90], [100, 110, 120]])
# x = x_input.reshape(-1, 3, 1)

# y_pred = model.predict(x)
print(res)
# print(loss2, mse2)
# print(y_pred)

# [0.21841164828779605, 0.15539617931846386, 0.06301546440674709, 0.15539617931846386, 0.06301546440674709]