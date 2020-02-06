from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

x_train = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_train = [0, 1, 1, 0]

model = LinearSVC()
model = SVC(kernel='poly', degree=2, gamma=1, coef0=0)
model = SVC(kernel='rbf', degree=2, gamma=1, coef0=0)
model = SVC(kernel='sigmoid', degree=2, gamma=1, coef0=0)
model = KNeighborsClassifier(n_neighbors=1)

model.fit(x_train, y_train)

y_pred = model.predict(x_train)

print(accuracy_score(y_train, y_pred))