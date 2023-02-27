import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import RANSACRegressor, HuberRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.svm import SVR

df = pd.read_csv('data/insurance.csv')

X = df.drop(['sex', 'children', 'charges'], axis=1)
X.smoker.replace(['no', 'yes'], [0, 1], inplace=True)
X.region.replace(['southwest', 'southeast', 'northwest', 'northeast'], [0, 1, 2, 3], inplace=True)
Y = df['charges']

"""models = {
    'RANSACR': RANSACRegressor(),
    'HubberR': HuberRegressor(),
    'SVR': SVR()
}

params = {
    'RANSACR': {},
    'HubberR': {
        'max_iter': [5000],
        'epsilon': [1.35]
    },
    'SVR': {
        'kernel': ['linear', 'poly', 'rbf'],
        'gamma': ['auto'],
        'C': [1, 10],
        'epsilon': [0.1]

    }
}"""

kfold = KFold(n_splits=10, random_state=7, shuffle=True)

"""best_score = 999
best_model = None

for key, value in models.items():
    grid = GridSearchCV(value, params[key], cv=kfold).fit(X, Y)
    score = np.abs(grid.best_score_)

    if score < best_score:
        best_score = score
        best_model = grid.best_estimator_"""

#print(best_score, best_model, sep='\n')

models = (
    ('Huber', HuberRegressor(max_iter=5000, epsilon=1.35)),
    ('RANSAC', RANSACRegressor())
)


scores = {}
for name, model in models:
    scores[name] = np.mean(cross_val_score(model, X, Y, scoring='r2', cv=kfold))

print(scores)