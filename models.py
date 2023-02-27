import pandas as pd
import numpy as np
import utils
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

class Models:
    def __init__(self):
        self.models = [('LiR', LinearRegression()),
                  ('CART', DecisionTreeRegressor()),
                  ('kNN', KNeighborsRegressor()),
                  ('SVR', SVR(gamma='auto', kernel='poly', C=10))]

    def models_results(self):

        X, Y = utils.Utils().get_x_y()
        X = StandardScaler().fit_transform(X)
        kfold = KFold(n_splits=10, random_state=7, shuffle=True)
        results = []

        for name, model in self.models:
            model_score = cross_val_score(model, X, Y, cv=kfold, scoring='r2')
            results.append((name, model_score.mean(), model_score.std()))

        return results

    def tuning_models(self, names_models, parameters):
        models = {i: j for i, j in self.models if i in names_models}
        grid = GridSearchCV(models, parameters)
