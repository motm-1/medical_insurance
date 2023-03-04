import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils as ut
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


def knn_plotting(algorithm, n_neighbors, p, weights):
    knn = KNeighborsRegressor(algorithm=algorithm, n_neighbors=n_neighbors, p=p, weights=weights)
    X, Y = ut.Utils().clean_x_y()
    X = ut.Utils().standard_scaler(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    knn.fit(X_train, Y_train)
    y_hat = knn.predict(X_test)

    print(f'MSE: {mean_squared_error(Y_test, y_hat)}\n'
          f'R2 {r2_score(Y_test, y_hat)}\n')

    sns.set()
    sns.kdeplot(Y_test, color='black', linestyle='dashed')
    sns.kdeplot(y_hat, color='blue')
    plt.show()