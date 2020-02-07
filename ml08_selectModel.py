import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.testing import all_estimators
import warnings
warnings.filterwarnings('ignore')

iris_data = pd.read_csv('data\iris2.csv', encoding='utf-8')

# print(iris_data.head(10))
# quit()
x = iris_data.iloc[:, :-1]
y = iris_data.iloc[:, [-1]]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
allAlgorithms = all_estimators(type_filter='classifier')

print(allAlgorithms)
print(len(allAlgorithms))
print(type(allAlgorithms))

for(name,algorithm) in allAlgorithms:
    clf = algorithm()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('acc : ', accuracy_score(y_test, y_pred))