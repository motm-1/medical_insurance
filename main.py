import models as md
import pandas as pd
from outliers import anti_outliers_models
from final_model import knn_plotting


def run():
    models = md.Models()
    results = models.models_results()
    print(results, '\n')
    algorithms_names = data_to_tune(results)
    dicts_tuning_info = tune_models(algorithms_names, models)
    count = 0
    print('\n', anti_outliers_models())

    for dictionary in dicts_tuning_info:
        df = pd.DataFrame(dictionary)
        df.to_csv(f'./data/{count}', index=False)
        count += 1


def plot_knn():
    knn_plotting(algorithm='kd_tree', n_neighbors=7, p=2, weights='uniform')


def data_to_tune(results):
    names, r2 = (
        [i for i, j, k in results],
        [j for i, j, k in results]
    )
    r2_copy = r2.copy()
    r2_copy.sort(reverse=True)

    algorithms_names = [names[r2.index(r2_copy[0])], names[r2.index(r2_copy[1])]]

    return algorithms_names


def tune_models(algorithms_names, models):
    parameters = {
        algorithms_names[0]: {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'algorithm': ['ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2]
        },
        algorithms_names[1]: {}
    }

    after_tuning_results = models.tuning_models(algorithms_names, parameters)

    print(f'Scores: {after_tuning_results["scores"]} \n'
          f'Estimators: {after_tuning_results["estimators"]}')

    return after_tuning_results["cv_results"]


if __name__ == "__main__":
    # run()
    plot_knn()
