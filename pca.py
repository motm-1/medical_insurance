import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score


if __name__ == "__main__":
    df = pd.read_csv('data/insurance.csv')
    X = df.drop(['charges'], axis=1)
    Y = df['charges']

    X.sex.replace(['male', 'female'], [0, 1], inplace=True)
    X.smoker.replace(['no', 'yes'], [0, 1], inplace=True)
    X.region.replace(['southwest', 'southeast', 'northwest', 'northeast'], [0, 1, 2, 3], inplace=True)

    X_2 = StandardScaler().fit_transform(X.drop(['sex', 'children'], axis=1))
    X = StandardScaler().fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=0.3,
        random_state=7,
        shuffle=True
    )
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)

    lir = LinearRegression()
    lir.fit(X_train, Y_train)
    no_pca_y = lir.predict(X_test)
    print(f'No PCA LiR R2: {r2_score(Y_test, no_pca_y)}')

    pca = PCA(n_components=3)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    lir.fit(X_train_pca, Y_train)
    pca_y = lir.predict(X_test_pca)
    print(f'PCA LiR R2: {r2_score(Y_test, pca_y)}')

    kpca = KernelPCA(n_components=3, kernel='poly')
    kpca.fit(X_train)
    X_train_kpca = kpca.transform(X_train)
    X_test_kpca = kpca.transform(X_test)
    lir.fit(X_train_kpca, Y_train)
    kpca_y = lir.predict(X_test_pca)
    print(f'KPCA LiR R2: {r2_score(Y_test, kpca_y)}')

    X2_train, X2_test, Y2_train, Y2_test = train_test_split(
        X_2, Y,
        test_size=0.3,
        random_state=7,
        shuffle=True
    )

    model = LinearRegression()
    model.fit(X2_train, Y2_train)
    y2_results = model.predict(X2_test)
    x2_results = cross_val_score(model, X_2, Y, cv=kfold, scoring='r2')
    print(f'Without SEX and CHILDREN columns LiR R2: {np.mean(x2_results)} \n')

    lir.fit(X_train, Y_train)
    y_lir = lir.predict(X_test)
    lasso = Lasso(alpha=0.01).fit(X_train, Y_train)
    y_lasso = lasso.predict(X_test)
    ridge = Ridge(alpha=1).fit(X_train, Y_train)
    y_ridge = ridge.predict(X_test)

    print(f'LiR MSE: {mean_squared_error(Y_test, y_lir)} \n'
          f'X2 LiR MSE {mean_squared_error(Y2_test, y2_results)} \n'
          f'Lasso MSE: {mean_squared_error(Y_test, y_lasso)} \n'
          f'Ridge MSE: {mean_squared_error(Y_test, y_ridge)} \n')

    print(f'Lasso COEFs {lasso.coef_} \n'
          f'Ridge Coefs {ridge.coef_}')

    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_, color='red')
    plt.show()

    sns.kdeplot(Y_test, color='black', linestyle='dashed')
    sns.kdeplot(no_pca_y, color='blue')
    sns.kdeplot(y2_results, color='grey')
    sns.kdeplot(pca_y, color='red')
    sns.kdeplot(kpca_y, color='purple')
    plt.show()