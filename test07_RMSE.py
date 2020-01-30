
from sklearn.model_selection import train_test_split
import numpy as np

model = Sequential()

# model.add(Dense(32, input_dim=1))
model.add(Dense(32, input_shape=(1, )))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1))
# model.fit(x, y, epochs=100)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=0)

print(len(x_train) / len(x))
print(len(x_val) / len(x))
print(len(x_test) / len(x))


model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=50, batch_size=1, validation_data=(x_val, y_val))

loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print('mse:', mse) 

x_pred = np.arange(100, 105, 1)
result = model.predict(x_pred, batch_size=1)
print(result)

from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE :', RMSE(y_test, y_pred))