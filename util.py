import numpy as np
import pandas as pd

def load_data():
    df = pd.read_csv('weatherHistory.csv')

    data = df.iloc[:,3:7].values

    temperature = data[:, 0:1]
    apparent_temperature = data[:,1:2]
    humidity = data[:, 2:3]
    windSpeed = data[:, 3:4]

    X_train = np.concatenate((temperature, humidity, windSpeed), axis=1)
    y_train = apparent_temperature.flatten()


    # print(f'temperatue =\n {temperature}')
    # print(f'\napparent_temperature =\n {apparent_temperature}')
    # print(f'\nhumidity =\n {humidity}')
    # print(f'\nwindSpeed =\n {windSpeed}')

    # print(f'\ny_train =\n {y_train}')
    # print(f'\nX_train =\n {X_train}')
    return X_train, y_train


def predict(x, w, b):
    return np.dot(x, w) + b


def compute_cost(X, y, w, b):
    m = X.shape[0]
    sum = 0
    for i in range(m):
        sum += (predict(X[i], w, b) - y[i]) ** 2

    return sum / (2 * m)


def compute_gradient(X, y, w, b):
    m = X.shape[0]
    n = X.shape[1]

    dj_db = 0
    dj_dw = np.zeros(n)

    for i in range(m):
        residual = predict(X[i], w, b) - y[i]
        dj_db += residual

        for j in range(n):
            dj_dw[j] += residual * X[i][j]

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    print('new')
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w_in, b_in)

        w_in = w_in - (alpha * dj_dw)
        b_in = b_in - (alpha * dj_db)

        if i % (num_iters / 10) == 0:
            print(f"Iteration   {i}: Cost  {compute_cost(X, y, w_in, b_in)}")

    return w_in, b_in