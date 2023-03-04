import pandas as pd
import numpy as np
from sklearn.linear_model import RANSACRegressor, HuberRegressor
from sklearn.model_selection import KFold, GridSearchCV
import utils

UTILS = utils.Utils()


def anti_outliers_models():
    X, Y = UTILS.clean_x_y()
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)

    models = {
        'Huber': HuberRegressor(),
        'RANSAC': RANSACRegressor()
    }
    parameters = {
        'Huber': {
            'epsilon': [1.35, 1, 3],
            'alpha': [0.0001, 0.001,  0.01, 0.1, 1],
            'max_iter': [5000]
        },
        'RANSAC': {
            'min_samples': [0.1, 0.3, 0.5, None]
        }
    }

    scores = []

    for key, value in models.items():
        grid = GridSearchCV(value, parameters[key], cv=kfold, scoring='r2').fit(X, Y)
        scores.append((key, grid.best_score_, grid.best_estimator_))

    return scores
