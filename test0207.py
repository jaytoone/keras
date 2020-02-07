# import pandas_datareader.data as wb
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


def min_max_scaling(x):
    x_np = np.asarray(x)
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7)  # 1e-7은 0으로 나누는 오류 예방차원


# 정규화된 값을 원래의 값으로 되돌린다
# 정규화하기 이전의 org_x값과 되돌리고 싶은 x를 입력하면 역정규화된 값을 리턴한다
def reverse_min_max_scaling(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()


#           Params          #
input_data_column_cnt = 6
output_data_column_cnt = 1
seq_length = 28
rnn_cell_hidden_dim = 20
forget_bias = 1.0
num_stacked_layers = 1
keep_prob = 1.0

epoch_num = 1000
learning_rate = 0.01

stock_file_name = 'samsung.xlsx'  # 아마존 주가데이터 파일
encoding = 'euc-kr'  # 문자 인코딩
columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
data = pd.read_excel(stock_file_name, encoding=encoding)  # 판다스이용 csv파일 로딩
raw_dataframe = pd.DataFrame(data, columns=columns)

del raw_dataframe['Date']
stock_info = raw_dataframe.values[:].astype(np.float)
print("stock_info.shape: ", stock_info.shape)

price = stock_info[:, :-1]
norm_price = min_max_scaling(price)

volume = stock_info[:, -1:]
norm_volume = min_max_scaling(volume)

x = np.concatenate((norm_price, norm_volume), axis=1)  # axis=1, 세로로 합친다
y = x[:, [-2]]
print("x.shape: ", x.shape)

dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i: i + seq_length]
    _y = y[i + seq_length]  # 다음 나타날 주가(정답)
    dataX.append(_x)  # dataX 리스트에 추가
    dataY.append(_y)  # dataY 리스트에 추가

Made_X = np.array(dataX)
Made_Y = np.array(dataY)
print(Made_X.shape, Made_Y.shape)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(Made_X, Made_Y, test_size=0.3, random_state=206)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=206)
print(x_test.shape)

#                   Data Preprocessing Completed                    #
#
# from keras.models import Sequential, Model
# from keras.layers import Dense, LSTM, Input, Flatten
#
# model = Sequential()
# model.add(Dense(100, activation='relu', input_shape=(28, 6)))
# model.add(LSTM(100, activation='relu'))
# # model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1))
# model.summary()
#
# #                   Model Training                      #
#
# from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
#
# filepath="model/samsung_stock.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# checkpoint2 = TensorBoard(log_dir='Tensorboard_graph',
#                           histogram_freq=0,
#                           write_graph=True,
#                           write_images=True)
# checkpoint3 = EarlyStopping(monitor='val_loss', patience=20)
# callback_list = [checkpoint, checkpoint2, checkpoint3]
# model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.fit(x_train, y_train, epochs=1000, validation_data=(x_val, y_val), verbose=2, callbacks=callback_list)
# acc = model.evaluate(x_val, y_val)
# print(acc[0])

# #           Load Model          #
x_train, x_test, y_train, y_test = train_test_split(Made_X, Made_Y, test_size=0.3, shuffle=False)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, shuffle=False)
from keras.models import load_model

model = load_model('model/samsung_stock.hdf5')

#          Prediction          #
# y_pred = model.predict(x_test, verbose=1)
price_today = model.predict(np.array(x[-seq_length:]).reshape(-1, 28, 6))
print(reverse_min_max_scaling(price, price_today))

#           오늘 종가 : 60150원          #

plt.plot(y_test, 'r')
plt.plot(y_pred, 'b')
plt.savefig('model/figure_pred.png')