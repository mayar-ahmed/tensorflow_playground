import numpy as np

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.uint8)
X_val = X_val.astype(np.float32)
y_val = y_val.astype(np.uint8)
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.uint8)

np.save("data/cifar10/X_train.npy", X_train)
np.save("data/cifar10/y_train.npy", y_train)
np.save("data/cifar10/X_val.npy", X_val)
np.save("data/cifar10/y_val.npy", y_val)
np.save("data/cifar10/X_test.npy", X_test)
np.save("data/cifar10/y_test.npy", y_test)

print('Train data shape: ', X_train.shape, " dtype ", X_train.dtype)
print('Train labels shape : ', y_train.shape, " dtype ", y_train.dtype)
print('Validation data shape: ', X_val.shape, " dtype ", X_val.dtype)
print('Validation labels shape: ', y_val.shape, " dtype ", y_val.dtype)
print('Test data shape: ', X_test.shape, " dtype ", X_test.dtype)
print('Test labels shape: ', y_test.shape, " dtype ", y_val.dtype)
