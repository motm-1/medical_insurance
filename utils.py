import pandas as pd


class Utils:
    def __init__(self):
        self.df = pd.read_csv('./data/insurance.csv')

    def get_x_y(self):
        X = self.df.drop(['charges'], axis=1)
        Y = self.df['charges']
        try:
            X.sex.replace(['male', 'female'], [0, 1], inplace=True)
            X.smoker.replace(['no', 'yes'], [0, 1], inplace=True)
            X.region.replace(['southwest', 'southeast', 'northwest', 'northeast'], [0, 1, 2, 3], inplace=True)
        except Exception as e:
            print(e)

        return X, Y

