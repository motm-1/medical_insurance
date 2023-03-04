import pandas as pd
import numpy as np
import utils
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


class Models:
    def __init__(self):
        self.models = [('LiR', LinearRegression()),
                       ('CART', DecisionTreeRegressor()),
                       ('kNN', KNeighborsRegressor()),
                       ('SVR', SVR(gamma='auto', kernel='poly', C=10))]

        self.kfold = KFold(n_splits=10, random_state=7, shuffle=True)

        self.X, self.Y = utils.Utils().clean_x_y()

        self.X = utils.Utils().standard_scaler(self.X)

    def models_results(self):
        results = []

        for name, model in self.models:
            model_score = cross_val_score(model, self.X, self.Y, cv=self.kfold, scoring='r2')
            results.append((name, model_score.mean(), model_score.std()))

        return results

    def tuning_models(self, names_models, parameters):
        models = {i: j for i, j in self.models if i in names_models}
        scores = []
        best_estimators = []
        info = []
        for key, value in models.items():
            grid = GridSearchCV(value, parameters[key], cv=self.kfold, scoring='r2').fit(self.X, self.Y)
            scores.append(grid.best_score_)
            best_estimators.append(grid.best_estimator_)
            info.append(grid.cv_results_)

        return {
            'scores': scores,
            'estimators': best_estimators,
            'cv_results': info
        }