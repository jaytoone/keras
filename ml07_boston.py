from sklearn.datasets import load_boston
boston = load_boston()

x = boston.data
y = boston.target

from sklearn.linear_model import LinearRegression, Ridge, Lasso

model = LinearRegression()
# model = Ridge()
# model = Lasso()

model.fit(x, y)

acc = model.score(x, y)
print('acc : ', acc)