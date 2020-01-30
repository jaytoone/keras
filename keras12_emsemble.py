from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Input


x = np.array([np.arange(1, 101), np.arange(101, 201), np.arange(301, 401)])
y = np.array([np.arange(1, 101)])

x2 = np.array([np.arange(1001, 1101), np.arange(1101, 1201), np.arange(1301, 1401)])
# y2 = np.array([np.arange(1001, 1101)])

# x3 = np.array([np.arange(1, 101), np.arange(101, 201), np.arange(301, 401)])
# y3 = np.array([np.arange(1, 101)])

x = x.reshape(-1, 3)
y = y.reshape(-1, 1)
x2 = x2.reshape(-1, 3)
print(x.shape, y.shape)

# x = np.transpose(x)

# concat_x = np.concatenate((x, x2), axis=1)

x_train, x_test, x2_train, x2_test, y_train, y_test = train_test_split(x, x2, y, test_size=0.4, random_state=0)
x_val, x_test, x2_val, x2_test, y_val, y_test = train_test_split(x_test, x2_test, y_test, test_size=0.5, random_state=0)

print(len(x2_train) / len(x2))
print(len(x2_val) / len(x2))
print(len(x2_test) / len(x2))
# quit()

# model = Sequential()
# model.add(Dense(32, input_shape=(3, )))
# model.add(Dense(128))
# model.add(Dense(32))
# model.add(Dense(1))

#   Functional API
input = Input(shape=(3,))
model = Dense(32)(input)
model = Dense(32)(model)
model = Dense(32)(model)
output = Dense(1)(model)
# model = Model(inputs=input, outputs=output)

#   Second Model for Ensemble
input2 = Input(shape=(3,))
model2 = Dense(64)(input2)
model2 = Dense(64)(model2)
model2 = Dense(64)(model2)
output2 = Dense(5)(model2) # merge 하는 경우 마지막 노드 수는 상관없다. hidden cell 이기 때문

from keras.layers.merge import concatenate

merge = concatenate([output, output2])

merge = Dense(4)(merge)
merge = Dense(4)(merge)
output = Dense(1)(merge)

#   merge 의 경우 model 을 정의하는 방법
model = Model(inputs=[input, input2], outputs=output)
model.summary()
# quit()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x_train, x2_train], y_train, epochs=100, batch_size=1, validation_data=([x_val, x2_val], y_val))

loss, mse = model.evaluate([x_test, x2_test], y_test, batch_size=1)
print('mse :', mse) 

y_pred = model.predict([x_test, x2_test], batch_size=1)
print(y_test.shape, y_pred.shape)
# quit()

from sklearn.metrics import mean_squared_error, r2_score

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE :', RMSE(y_test, y_pred))

R2_score = r2_score(y_test, y_pred)
print('R2_score :', R2_score)