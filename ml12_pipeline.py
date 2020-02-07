import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.utils.testing import all_estimators
import warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')

iris_data = pd.read_csv('data\iris2.csv', encoding='utf-8')

# print(iris_data.head(10))
# quit()
x = iris_data.iloc[:, :-1]
y = iris_data.iloc[:, [-1]]
print(x.shape, y.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
allAlgorithms = all_estimators(type_filter='classifier')
# quit()

kfold_cv = KFold(n_splits=5, shuffle=True)

print(allAlgorithms)
print(len(allAlgorithms))
print(type(allAlgorithms))

# for(name,algorithm) in allAlgorithms:
#     clf = algorithm()
#     if hasattr(clf, 'score'):
#         scores = cross_val_score(clf, x, y, cv=kfold_cv)
#         print(scores, end=' ')
#         print(name)
        

parameters = [
    {'C': [1, 10, 100, 100], 'kernel': ['linear']},
    {'C': [1, 10, 100, 100], 'kernel': ['rbf'], 'gamma': [0.001, 0.0001]},
    {'C': [1, 10, 100, 100], 'kernel': ['sigmoid'], 'gamma': [0.001, 0.0001]}
]

kfold_cv = KFold(n_splits=5, shuffle=True)
# pipe = Pipeline([("classifier", SVC())])
pipe = Pipeline([('scaler', MinMaxScaler()), ('svm', SVC())])

#           아직 RandomizedSearchCV 해결 못함 !             #
# parameters = {'C': [1, 10, 100, 100], 'kernel': ['linear', 'rbf']}
# model = RandomizedSearchCV(pipe, param_distributions=parameters, n_iter=8, cv=kfold_cv)
# # model = GridSearchCV(SVC(), parameters, cv=kfold_cv)
# model.fit(x_train, y_train)

# print('최적 매개 변수 : ', model.best_estimator_)

# y_pred = model.predict(x_test)
# print('최종 정답률 : ', accuracy_score(y_test, y_pred))

pipe.fit(x_train, y_train)
print(pipe.score(x_test, y_test))