from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

x_train = [[0, 0], [1, 0], [0, 0], [1, 1]]
y_train = [0, 0, 0, 1]

model = LinearSVC()

model.fit(x_train, y_train)

y_pred = model.predict(x_train)

print(accuracy_score(y_train, y_pred))