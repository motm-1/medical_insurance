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
    parameters = {
        names_to_tune[0]: '',
        names_to_tune[1]: ''
    }
    after_tuning_results = models.tuning_models(names_to_tune, parameters)


if __name__ == "__main__":
    run()
