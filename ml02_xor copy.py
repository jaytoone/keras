from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


x_train = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y_train = np.array([0, 1, 1, 0])
print(x_train.shape, y_train.shape)

model = LinearSVC()
model = SVC(kernel='poly', degree=2, gamma=1, coef0=0)
model = SVC(kernel='rbf', degree=2, gamma=1, coef0=0)
model = SVC(kernel='sigmoid', degree=2, gamma=1, coef0=0)
model = KNeighborsClassifier(n_neighbors=1)

from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

model = Sequential()
model.add(Dense(16, activation='sigmoid', input_shape=(2, )))
model.add(Dense(1))
model.summary()

# quit()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train)

y_pred = model.predict(x_train)
print(y_pred)

# print(accuracy_score(y_train, y_pred))