
from sklearn.model_selection import train_test_split
import numpy as np


x = np.arange(1, 101)
y = np.arange(1, 101)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=0)

print(len(x_train) / len(x))
print(len(x_val) / len(x))
print(len(x_test) / len(x))