import models as md
import pandas as pd


def run():
    models = md.Models()
    results = models.models_results()
    names, r2 = (
        [i for i, j, k in results],
        [j for i, j, k in results]
    )
    r2_copy = r2.copy()
    r2_copy.sort(reverse=True)

    names_to_tune = [names[r2.index(r2_copy[0])], names[r2.index(r2_copy[1])]]
    print(names_to_tune)

    def tune_models():
        parameters = {
            names_to_tune[0]: {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance'],
                'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                'p': [1, 2]
            },
            names_to_tune[1]: {}
        }

        after_tuning_results = models.tuning_models(names_to_tune, parameters)

        print(f'Scores: {after_tuning_results["scores"]} \n'
              f'Estimators: {after_tuning_results["estimators"]}')

        return after_tuning_results["cv_results"]

    return pd.DataFrame(tune_models())


if __name__ == "__main__":
    run().to_csv('./data/analysis_data.csv')