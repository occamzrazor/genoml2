import pathlib
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import tqdm
from sklearn import (
    dummy,
    feature_selection,
    linear_model,
    metrics,
    model_selection,
    pipeline,
)
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

RANDOM_STATE = 42
GENERAL_VERBOSITY = 1
LOG_REG_VERBOSITY = 0


class LogRegExperiment(object):
    def __init__(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        test_x: Optional[np.ndarray] = None,
        test_y: Optional[np.ndarray] = None,
    ):
        self.train_x = train_x
        self.train_y = train_y

        self.test_x = test_x
        self.test_y = test_y

        self.results: Optional[pd.DataFrame] = None

        self.param_grid = {
            "classify__C": [1e-5, 1e-4, 1e-3, 1e-2],
            "classify__l1_ratio": [0, 0.1, 0.2],
        }
        self.cv_count = 5
        self.max_iter = 5000
        self.scoring = "balanced_accuracy"
        self._init_pipeline()
        self._init_model()

    def _init_pipeline(self):
        self.pipeline = pipeline.Pipeline(
            [
                (
                    "classify",
                    linear_model.LogisticRegression(
                        penalty="elasticnet",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        solver="saga",
                        max_iter=self.max_iter,
                        verbose=LOG_REG_VERBOSITY,
                        warm_start=False,
                        n_jobs=-2,
                    ),
                )
            ],
            memory="models_cache",
            verbose=True,
        )

    def _init_model(
        self,
    ):
        sss = model_selection.StratifiedShuffleSplit(
            n_splits=self.cv_count, test_size=0.1
        )
        self.model = model_selection.GridSearchCV(
            self.pipeline,
            n_jobs=1,  # We do not double parallelize
            param_grid=self.param_grid,
            scoring=self.scoring,
            verbose=GENERAL_VERBOSITY,
            cv=sss,
        )

    @classmethod
    def from_data(cls, data_path) -> "LogRegExperiment":
        data = np.load(data_path)
        return cls(
            data["train_X"], data["train_y"], data.get("test_X"), data.get("test_y")
        )

    @classmethod
    def load_experiment(cls, data_dir) -> "LogRegExperiment":
        data_dir = pathlib.Path(data_dir)

        data = np.load(data_dir.joinpath("data.npz"))
        with open(data_dir.joinpath("model.joblib"), "rb") as fo:
            logreg_model = joblib.load(fo)

        self = cls(
            data["train_X"], data["train_y"], data.get("test_X"), data.get("test_y")
        )
        self.model = logreg_model
        self.score_model()
        return self

    def save_experiment(self, directory):
        directory = pathlib.Path(directory)
        directory.mkdir(exist_ok=False)

        with open(directory.joinpath("logreg_model.joblib"), "wb") as fi:
            joblib.dump(self.model, fi)

        if self.results is not None:
            self.results.to_csv(directory.joinpath("results.tsv"), sep="\t")

        data_file = directory.joinpath("data.npz")
        if self.test_x is not None:
            np.savez(
                data_file,
                train_X=self.train_x,
                train_y=self.train_y,
                test_X=self.test_x,
                test_y=self.test_y,
            )
        else:
            np.savez(data_file, train_X=self.train_x, train_y=self.train_y)

    def train_model(self, refit=False) -> None:
        fit = True
        try:
            check_is_fitted(self.model)
        except NotFittedError:
            fit = False
        except Exception as e:
            raise e
        if not fit or refit:
            self.model.fit(self.train_x, self.train_y)
        self.score_model()

    def score_model(self):
        if self.test_y is None:
            raise Exception("CANNOT SCORE WITH NO TEST")
        scores = dict()
        scores["LogReg"] = _score_model(self.model, self.test_x, self.test_y)

        scores.update(self._score_dummy())
        self.results = pd.DataFrame(scores)
        return self.results

    def _score_dummy(self) -> Dict:
        scores = dict()
        for strategy in ["most_frequent", "stratified", "prior", "uniform"]:
            dummy_cls = dummy.DummyClassifier(
                strategy=strategy, random_state=RANDOM_STATE
            )
            dummy_cls.fit(self.train_x, self.train_y)
            score = _score_model(dummy_cls, self.test_x, self.test_y)

            scores[strategy + "_score"] = score

        return scores


def _score_model(clf, test_x, test_y) -> Dict[str, float]:
    score = metrics.balanced_accuracy_score(test_y, clf.predict(test_x))
    return {"balanced_accuracy": score}


class TopKSelectorsExperiment(LogRegExperiment):
    def __init__(self, *args, ks: Optional[List[int]] = None, **kwargs):
        super().__init__(*args, **kwargs)

        if ks is None:
            ks = [100, 1000, 10000]
        ks = [min(self.train_x.shape[1], k) for k in ks]
        self.ks = list(set(ks))
        self.param_grid["reduce_dim__k"] = self.ks
        self.selector_model = feature_selection.SelectKBest(
            feature_selection.chi2, k=max(self.ks)
        )
        self.pipeline.steps.insert(0, ("reduce_dim", self.selector_model))


def main():
    data_path = pathlib.Path("data/pre-plinked")
    train_test_split = data_path.joinpath("train_test_split.npz")
    tks = TopKSelectorsExperiment.from_data(train_test_split)
    tks.train_model(refit=False)

    experiment_dir = pathlib.Path("data/logistic_regression_experiments")
    tks.save_experiment(experiment_dir)


if __name__ == "__main__":
    main()
    print("Completed!")
