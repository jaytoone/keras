from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

iris_data = pd.read_csv('./data/iris.csv', encoding='utf-8', names=['a', 'b', 'c', 'd', 'y'])

x = iris_data.loc[:, ['a', 'b', 'c', 'd']]
y = iris_data.loc[:, 'y']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.7, shuffle=True)

model = SVC()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print('ACC :', accuracy_score(y_test, y_pred))


from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input
from keras.utils import np_utils


def categori(x):
    if x == 'Iris-setosa':
        x = 0
    elif x == 'Iris-versicolor':
        x = 1
    elif x == 'Iris-virginica':
        x = 2
    return x
y_train = np.array(list(map(categori, y_train)))
y_test = np.array(list(map(categori, y_test)))
# print(y_train)
# print(y_train.shape)
# quit()
y_train = np_utils.to_categorical(y_train, 4)
y_test = np_utils.to_categorical(y_test, 4)
# print(y_train.shape)
# quit()

model = Sequential()
model.add(Dense(12, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(x_train, y_train, epochs=100, batch_size=10)

print('ACC : %.4f' % (model.evaluate(x_test, y_test)[1]))