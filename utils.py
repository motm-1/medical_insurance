import pandas as pd
from sklearn.preprocessing import StandardScaler


class Utils:
    def __init__(self):
        self.df = pd.read_csv('./data/insurance.csv')

        self.X = self.df.drop(['charges'], axis=1)

        self.Y = self.df['charges']

    def clean_x_y(self):

        try:
            self.X.sex.replace(['male', 'female'], [0, 1], inplace=True)
            self.X.smoker.replace(['no', 'yes'], [0, 1], inplace=True)
            self.X.region.replace(['southwest', 'southeast', 'northwest', 'northeast'], [0, 1, 2, 3], inplace=True)
        except Exception as e:
            print(e)

        return self.X, self.Y

    def standard_scaler(self, X):

        X = StandardScaler().fit_transform(X)

        return X
