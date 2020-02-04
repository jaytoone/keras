import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


#       Params      #
input_data_length = 28



def min_max_scaler(price):
    Scaler = MinMaxScaler()
    Scaler.fit(price)

    return Scaler.transform(price)


ohlcv_csv = pd.read_csv('./data/samsung.csv')
ohlcv_csv2 = pd.read_csv('./data/kospi200.csv')
# print(ohlcv_csv.iloc[:, 1:])
# quit()
# print(ohlcv_csv.columns)
# quit()

def Make_X(ohlcv_csv):
    
    for column in ohlcv_csv.columns[1:]:
        try:
            ohlcv_csv[column] = ohlcv_csv[column].map(lambda x: int(x.replace(',', '')))
            
        except Exception as e:
            continue
        
    ohlcv_csv = ohlcv_csv.sort_index(ascending=False)
    # print(ohlcv_csv)
    # quit()

    ohlcv_data = ohlcv_csv.values[:, 1:].astype(np.float)
    # print(ohlcv_data)
    # quit()

    #           데이터 리스케일링           #
    price = ohlcv_data[:, :4]
    volume = ohlcv_data[:, [4]]

    #   Flexible Y_data / 자른 열데이터의 다음 데이터    #
    y = ohlcv_data[:, [3]]

    scaled_price = min_max_scaler(price)
    scaled_volume = min_max_scaler(volume)

    x = np.concatenate((scaled_price, scaled_volume), axis=1)


    dataX = []
    dataY = []
    #           grouping        #
    for i in range(input_data_length, len(ohlcv_data)):
        group_x = x[i - input_data_length: i]  # group_y 보다 1개 이전 데이터
        group_y = y[i]
        # print(group_x.shape)  # (28, 6)
        # print(group_y.shape)  # (1,)
        # quit()
        dataX.append(group_x)  # dataX 리스트에 추가
        dataY.append(group_y)  # dataY 리스트에 추가
        
    Made_X = np.array(dataX)
    Made_Y = np.array(dataY)
    
    return Made_X, Made_Y
# print(Made_X.shape) # (398, 28, 5)
# print(Made_Y.shape) # (398, 1)
# quit()

Made_X, Made_Y = Make_X(ohlcv_csv)
Made_X2, Made_Y2 = Make_X(ohlcv_csv2)
# print(Made_X2.shape)
# quit()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(Made_X, Made_Y, test_size=0.3, random_state=0)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=0)

x2_train, x2_test, y2_train, y2_test = train_test_split(Made_X2, Made_Y2, test_size=0.3, random_state=0)
x2_val, x2_test, y2_val, y2_test = train_test_split(x2_test, y2_test, test_size=0.5, random_state=0)

# print(x_train.shape) # (278, 28, 5)
# print(x_val.shape) # (60, 28, 5)
# print(x_test.shape) # (60, 28, 5)


#               Modeling            #

from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Dropout, LSTM
from keras.layers.merge import concatenate

model_num = int(input('Press model number : '))
#       Model types     #
#       1. DNN          #
if model_num == 1:
    input = Input(shape=(28, 5))
    model = Dense(64)(input)
    model = Dense(128)(model)
    # model = Dense(32)(model)
    drop_out = Dropout(0.3)(model)
    flatten = Flatten(name = 'flatten')(drop_out)
    output = Dense(1)(flatten)
    model = Model(inputs=input, outputs=output)

#       2. LSTM         #
elif model_num == 2:
    
    model = Sequential()
    model.add(LSTM(32, activation='relu', input_shape=(input_data_length, 5)))
    model.add(Dense(64))
    model.add(Dense(1))

#       3. DNN ENSEMBLE     #
elif model_num == 3:
    input = Input(shape=(28, 5))
    model = Dense(64)(input)
    model = Dense(128)(model)
    # model = Dense(32)(model)
    drop_out = Dropout(0.3)(model)
    flatten = Flatten(name = 'flatten')(drop_out)
    output = Dense(1)(flatten)
    
    input2 = Input(shape=(28, 5))
    model2 = Dense(32)(input2)
    model2 = Dense(32)(model2)
    # model = Dense(32)(model)
    drop_out2 = Dropout(0.3)(model2)
    flatten2 = Flatten(name = 'flatten2')(drop_out2)
    output2 = Dense(1)(flatten2)
    
    merge = concatenate([output, output2])
    output3 = Dense(1)(merge)
    model = Model(inputs=[input, input2], outputs=output3)

#       4. LSTM ENSEMBLE     #
elif model_num == 4:
    model = Sequential()
    model.add(LSTM(32, activation='relu', input_shape=(input_data_length, 5)))
    model.add(Dense(64))
    model.add(Dense(1))
    
    model2 = Sequential()
    model2.add(LSTM(32, activation='relu', input_shape=(input_data_length, 5)))
    model2.add(Dense(64))
    model2.add(Dense(1))
    
    merge = concatenate([model.output, model2.output])
    output3 = Dense(1)(merge)
    model = Model(inputs=[model.input, model2.input], outputs=output3)

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
filepath="./model/stock_prediction %s_%s.hdf5" % (input_data_length, model_num)
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
checkpoint2 = TensorBoard(log_dir='Tensorboard_graph',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)
checkpoint3 = EarlyStopping(monitor='val_loss', patience=15)
callbacks_list = [checkpoint, checkpoint2, checkpoint3]

model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

if model_num in [1, 2]:
    model.fit(x_train, y_train, epochs=200, batch_size=1, callbacks=callbacks_list, validation_data=(x_val, y_val))
else:
    model.fit([x_train, x2_train], y_train, epochs=200, batch_size=1, callbacks=callbacks_list, validation_data=([x_val, x2_val], y_val))

# from keras.models import load_model

# model = load_model('./model/stock_prediction 28_1.hdf5')

from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

if model_num in [1, 2]:
    y_pred = model.predict(x_test)
else:
    y_pred = model.predict([x_test, x2_test])        
    
print('RMSE :', RMSE(y_pred, y_test))
