import numpy as np
import calculator as calc


def load_data(file_x, file_y):
    X = np.loadtxt(file_x, delimiter=" ")
    Y = np.loadtxt(file_y, delimiter=" ")
    return X.T, Y.T


def initialize_parameters(layer_dims, seed=None):
    """
    :param layer_dims: レイヤーごとのニューロンのサイズを入れておく
    :return: parameters を返す。この中には W, b がそれぞれ入ってる
    """
    if seed is not None:
        np.random.seed(seed)

    L = len(layer_dims)
    parameters = {}
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def activation_forward(Z, activation):
    if activation == "sigmoid":
        A = calc.sigmoid(Z)
        activation_cache = Z
        return A, activation_cache
    elif activation == "relu":
        A = calc.relu(Z)
        activation_cache = Z
        return A, activation_cache


def linear_forward(A_prev, W, b):
    """
    アクティベーションの前段階のZを求める
    :param A_prev: 一つ前のニューロンの結果
    :param W: アクティベート用のW
    :param b: バイアス
    :return: Z, cache(バックプロパゲーションよう)
    """
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    一つのレイヤーのforward propagationを計算
    :param A_prev: 一つ前のニューロンの結果
    :param W: アクティベートW
    :param b: バイアス
    :param activation: アクティベーションに使う関数名
    :return: アクティベーションの結果, cache(バックプロパゲーション用)
    """
    Z, linear_cache = linear_forward(A_prev, W, b)
    A, activation_cache = activation_forward(Z, activation)
    cache = (linear_cache, activation_cache)
    return A, cache


def forward_propagation(X, parameters, activations):
    """
    L-modelのフォワードプロパゲーションを計算
    :param X: input の値
    :param parameters:  それぞれのアクティベーションに使うパラメータ
    :param activations: それぞれのレイヤーで使う非線形アクティベート関数
    :return: AL(結果), caches(バックプロパゲーション用)
    """
    L = len(parameters) // 2
    A = X
    caches = []
    for l in range(1, L):
        A_prev = A
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        A, cache = linear_activation_forward(A_prev, W, b, activations[l])
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activations[L])
    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL))) / m
    cost = np.squeeze(cost)
    return cost


def linear_back_propagation(dZ, cache):
    m = dZ.shape[1]
    A_prev, W, b = cache
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    da = np.dot(W.T, dZ)
    return da, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "sigmoid":
        sigmoid_backward = calc.sigmoid_derivative(activation_cache)
        dZ = np.multiply(dA, sigmoid_backward)
        dA_prev, dw, db = linear_back_propagation(dZ, linear_cache)
        return dA_prev, dw, db
    elif activation == "relu":
        relu_backward = calc.relu_derivative(activation_cache)
        dZ = np.multiply(dA, relu_backward)
        dA_prev, dw, db = linear_back_propagation(dZ, linear_cache)
        return dA_prev, dw, db


def back_propagation(AL, Y, caches, activations):
    grads = {}
    L = len(caches)
    dA_prev = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    for l in reversed(range(L)):
        cache = caches[l]
        dA_prev, dW, db = linear_activation_backward(dA_prev, cache, activations[l+1])
        grads["dA" + str(l + 1)] = dA_prev
        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db

    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters)
    for l in range(1, L + 1):
        parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate * grads["db" + str(l)]
    return parameters


# print("Load data")
# X, Y = load_data("data/train/X.txt", "data/train/Y.txt")
# activations = ["no use", "relu", "relu", "relu", "relu", "sigmoid"]
# layer_dims = [64, 100, 150, 100, 30, 65]
#
# print("initialize parameters")
# parameters = initialize_parameters(layer_dims)
#
# print("forward propagation")
# AL, caches = forward_propagation(X, parameters, activations)
#
# print("compute cost")
# cost = compute_cost(AL, Y)
# print(cost)
#
# print("backward propagation")
# grads = back_propagation(AL, Y, caches, activations)
