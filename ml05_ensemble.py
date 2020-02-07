import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

wine = pd.read_csv('data\winequality-white.csv', sep=';', encoding='utf-8')

x = wine.drop('quality', axis=1)
y = wine.quality

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

model = RandomForestClassifier()

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

y_pred = model.predict(x_test)
acc2 = accuracy_score(y_test, y_pred)



from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train, 11)
y_test = np_utils.to_categorical(y_test, 11)

model = Sequential()
model.add(Dense(100, input_dim=11, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(11, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from keras.callbacks import EarlyStopping
callback1 = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.fit(x_train, y_train, epochs=1000, batch_size=10)


print(acc)

