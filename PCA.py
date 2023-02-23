import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
    print(f'No PCA LiR Score: {lir.score(X_test, Y_test)}')

    pca = PCA(n_components=3)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    lir.fit(X_train_pca, Y_train)
    pca_y = lir.predict(X_test_pca)
    print(f'PCA LiR Score: {lir.score(X_test_pca, Y_test)}')
    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    plt.show()

    sns.kdeplot(Y_test, color='black', linestyle='dashed')
    sns.kdeplot(no_pca_y, color='blue')
    sns.kdeplot(pca_y, color='red')
    plt.show()
