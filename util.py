import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv('weatherHistory.csv')

    data = df.iloc[:,3:7].values

    temperature = data[:, 0:1]
    apparent_temperature = data[:,1:2]
    humidity = data[:, 2:3]
    windSpeed = data[:, 3:4]

    X_train = np.concatenate((temperature, humidity, windSpeed), axis=1)
    y_train = apparent_temperature.flatten()

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

        dj_dw += residual * X[i]
        
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

def stochastic_gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    print('new')
    for i in range(num_iters):
        rand = np.random.randint(X.shape[0])

        residual = predict(X[rand], w_in, b_in) - y[rand]

        dj_db = residual
        dj_dw =  residual * X[rand]

        w_in = w_in - (alpha * dj_dw)
        b_in = b_in - (alpha * dj_db)

        if i % (num_iters / 10) == 0:
            print(f"Iteration   {i}: Cost  {compute_cost(X, y, w_in, b_in)}")

    return w_in, b_in

def target_prediction_plot(X, Y, w, b, values):
    test_amnt = values

    x_coord = np.arange(test_amnt)

    # prediction of first test_amnt examples
    f_wb = np.dot(w,np.transpose(X[:test_amnt])) + b

    plt.scatter(x_coord, Y[:test_amnt], label='target')
    plt.scatter(x_coord, f_wb, label='predict')
    plt.xlabel('# Examples')
    plt.legend(loc='best')
    plt.show()