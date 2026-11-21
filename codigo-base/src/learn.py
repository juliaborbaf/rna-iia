import numpy as np
np.random.seed(42)

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def load_mnist():
    # Load MNIST data from sklearn
    mnist = fetch_openml('mnist_784', as_frame=False, cache=True, version=1)
    X, y = mnist["data"], mnist["target"].astype(int)

    # Normalize the data
    X = X / 255.0

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # One-hot encode the labels
    encoder = OneHotEncoder(sparse_output=False)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test = encoder.transform(y_test.reshape(-1, 1))

    return X_train, X_test, y_train, y_test

def initialize_parameters(input_size, hidden_size, output_size):
    # pesos pequenos e aleatórios (quebram a simetria entre neurônios);
    # vieses zerados e 2D (1 x n) para casar com db1/db2 do backward.
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

# Activation functions
def relu(Z):
    # max(0, z) elemento a elemento
    # preserva o formato de Z
    return np.maximum(0, Z)

def relu_derivative(Z):
    return Z > 0

def softmax(Z):
    # Subtrai o máximo de cada linha antes da exponencial para evitar overflow.
    # O resultado não muda, pois softmax(z - c) = softmax(z).
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

# Forward pass
def forward(X, parameters):
    pass

# Loss function
def cross_entropy_loss(y_true, y_pred):
    pass

# Backward pass
def backward(X, y, W2, Z1, A1, A2):
    n_samples = X.shape[0]
    
    dZ2 = A2 - y
    dW2 = np.dot(A1.T, dZ2) / n_samples
    db2 = np.sum(dZ2, axis=0, keepdims=True) / n_samples
    dA1 = np.dot(dZ2, W2.T)
    
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / n_samples
    db1 = np.sum(dZ1, axis=0, keepdims=True) / n_samples
    
    return dW1, db1, dW2, db2

def gradient_descent_step(X, y, parameters, learning_rate=0.01):
    pass

def accuracy(y_true, y_pred):
    pass
    
    

    










