import numpy as np  

x = np.array(range(1, 21))
y = np.array(range(1, 21))  

x = x.reshape(20, 1)

print(x.shape, y.shape)     

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=231, shuffle=False)

print('             ')  
print(train_x)
print('             ')  
print(test_x)
print('             ')  

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

std_scaler = StandardScaler()
std_scaler.fit(x)
train_x = std_scaler.transform(train_x)
test_x = std_scaler.transform(test_x)

print('             ')  
print(train_x)
print('             ')  
print(test_x)
print('             ')  