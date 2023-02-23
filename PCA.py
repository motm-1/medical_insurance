import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    df = pd.read_csv('./insurance.csv')
    X = df.drop(['charges'], axis=1)
    Y = df['charges']

    X.sex.replace(['male', 'female'], [0, 1], inplace=True)
    X.smoker.replace(['no', 'yes'], [0, 1], inplace=True)
    X.region.replace(['southwest', 'southeast', 'northwest', 'northeast'], [0, 1, 2, 3], inplace=True)

    X = StandardScaler().fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=0.3,
        random_state=7,
        shuffle=True
    )

    lir = LinearRegression()
    lir.fit(X_train, Y_train)
    no_pca_y = lir.predict(X_test)
    print(f'No PCA LiR MSE: {mean_squared_error(Y_test, no_pca_y)}')

    pca = PCA(n_components=3)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    lir.fit(X_train_pca, Y_train)
    pca_y = lir.predict(X_test_pca)
    print(f'PCA LiR MSE: {mean_squared_error(Y_test, pca_y)}')

    kpca = KernelPCA(n_components=3, kernel='poly')
    kpca.fit(X_train)
    X_train_kpca = kpca.transform(X_train)
    X_test_kpca = kpca.transform(X_test)
    lir.fit(X_train_kpca, Y_train)
    kpca_y = lir.predict(X_test_pca)
    print(f'KPCA LiR MSE: {mean_squared_error(Y_test, kpca_y)} \n')

    lir.fit(X_train, Y_train)
    y_lir = lir.predict(X_test)
    lasso = Lasso(alpha=0.01).fit(X_train, Y_train)
    y_lasso = lasso.predict(X_test)
    ridge = Ridge(alpha=1).fit(X_train, Y_train)
    y_ridge = ridge.predict(X_test)

    print(f'LiR MSE: {mean_squared_error(Y_test, y_lir)} \n'
          f'Lasso MSE: {mean_squared_error(Y_test, y_lasso)} \n'
          f'Ridge MSE: {mean_squared_error(Y_test, y_ridge)} \n')

    print(f'Lasso COEFs {lasso.coef_} \n'
          f'Ridge Coefs {ridge.coef_}')

    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_, color='red')
    plt.show()

    sns.kdeplot(Y_test, color='black', linestyle='dashed')
    sns.kdeplot(no_pca_y, color='blue')
    sns.kdeplot(pca_y, color='red', legend=False)
    sns.kdeplot(kpca_y, color='purple')
    plt.show()
