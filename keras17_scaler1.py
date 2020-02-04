import numpy as np    
from keras.models import Model
from keras.layers import Dense, LSTM, Input

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [ 8,  9, 10],
           [ 9, 10, 11], [10, 11, 12], [2, 3, 4], [3, 4, 5], [4, 5, 6]])

y = np.array(np.arange(13)).reshape(-1, 1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

std_scaler = StandardScaler()
std_scaler.fit(x)
x = std_scaler.transform(x)
# std_scaler.fit(y)
# y = std_scaler.transform(y)
x = x.reshape(-1, 3, 1)

print(x.shape, y.shape)

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=231)

from keras.models import Model
from keras.layers import Dense, Input

inputs = Input(shape=(3, 1))
# model = LSTM(10)(inputs)
model = Dense(10)(inputs)
model = LSTM(10)(model)


model = Dense(5)(model)
output = Dense(1)(model)
model = Model(inputs=inputs, outputs=output)

model.summary()

from keras.callbacks import EarlyStopping

callback_list = EarlyStopping(monitor='loss', patience=20, mode='auto')

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x, y, epochs=1000, batch_size=1, callbacks=[callback_list])
loss, mae = model.evaluate(x, y, batch_size=1)

print(loss, mae)

from sklearn.metrics import mean_squared_error, r2_score

pred_y = model.predict(test_x)

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(test_y, pred_y))
print('RMSE :', RMSE(test_y, pred_y))
print('r2_score :', r2_score(test_y, pred_y))

# 0.8776823486774586 0.7603426323487208
# RMSE : 1.007945303413951
# r2_score : -0.01595373467424177

# 12.282660446989421 2.8304943304795485
# RMSE : 3.7690549023638793
# r2_score : -0.014698204073799293