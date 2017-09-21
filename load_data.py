import numpy as np

X_train=np.load("X_train.npy")
y_train=np.load("y_train.npy")
X_val=np.load("X_val.npy")
y_val=np.load("y_val.npy")
X_test=np.load("X_test.npy")
y_test=np.load("y_test.npy")

print('Train data shape: ',X_train.shape , " dtype ",X_train.dtype)
print('Train labels shape : ', y_train.shape ," dtype ", y_train.dtype)
print('Validation data shape: ', X_val.shape ," dtype ", X_val.dtype )
print('Validation labels shape: ', y_val.shape, " dtype ", y_val.dtype)
print('Test data shape: ', X_test.shape, " dtype ", X_test.dtype)
print('Test labels shape: ', y_test.shape , " dtype ", y_val.dtype)