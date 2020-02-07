import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.testing import all_estimators
import warnings
warnings.filterwarnings('ignore')

input_data_length = 54
model_num = input('Press model num : ')

Made_X = np.load('data/Made_X %s_%s.npy' % (input_data_length, model_num))
Made_Y = np.load('data/Made_Y %s_%s.npy' % (input_data_length, model_num))


x_train, x_test, y_train, y_test = train_test_split(Made_X, Made_Y, test_size=0.3, shuffle=True)
allAlgorithms = all_estimators(type_filter='classifier')

print(allAlgorithms)
print(len(allAlgorithms))
print(type(allAlgorithms))

for(name,algorithm) in allAlgorithms:
    clf = algorithm()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(accuracy_score(y_test, y_pred), name)