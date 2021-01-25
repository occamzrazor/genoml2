import pathlib
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
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

from genoml.preprocessing import plink2_reader

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
        _cache_dir="models_cache",
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
        self._cache_dir = _cache_dir
        self._init_pipeline(self._cache_dir)
        self._init_model()

    def _init_pipeline(self, _cache_dir: str):
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
            memory=_cache_dir,
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
            refit=self._decision_function,
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

        data_path = data_dir.joinpath("data.npz")
        if not data_path.exists():
            data = {
                "train_X": np.array([], ndmin=2),
                "train_y": np.array([], ndmin=2),
            }
        else:
            data = np.load(data_path)

        with open(data_dir.joinpath("model.joblib"), "rb") as fo:
            logreg_model = joblib.load(fo)

        self = cls(
            data["train_X"], data["train_y"], data.get("test_X"), data.get("test_y")
        )
        self.model = logreg_model
        self.pipeline = logreg_model.estimator

        if self.test_x is not None:
            self.score_model()
        return self

    def save_experiment(self, directory):
        directory = pathlib.Path(directory)
        directory.mkdir(exist_ok=False)

        with open(directory.joinpath("model.joblib"), "wb") as fi:
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

    def train_model(self, completely_refit=False) -> None:
        self._train_model(self.model, self.train_x, self.train_y, completely_refit)
        self.score_model()

    @staticmethod
    def _train_model(model, x, y, completely_refit: bool = False):
        fit = True
        try:
            check_is_fitted(model)
        except NotFittedError:
            fit = False
        except Exception as e:
            raise e
        if not fit or completely_refit:
            model.fit(x, y)
        return model

    def score_model(self):
        if self.test_y is None:
            raise Exception("CANNOT SCORE WITH NO TEST")
        scores = dict()
        scores["LogReg"] = _score_model(self.model, self.test_x, self.test_y)

        scores.update(self._score_dummy())
        self.results = pd.DataFrame(scores).T
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

    @staticmethod
    def _decision_function(cv_results_) -> int:
        """Helps the gridsearchCV decide on the best model to refit.

        :param cv_results_: dict of numpy arrays. Comes directly from the gridsearchCV,
        :return: The best model's index.
        """
        results = pd.DataFrame(cv_results_)
        results["score_col"] = results["mean_test_score"] - results["std_test_score"]
        results = results.sort_values(by="score_col", ascending=False)
        return results.index[0]


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

    def extract_top_k_features(
        self, original_pvar_file, to_csv_file: Optional[str] = None
    ) -> pd.DataFrame:
        """Extracts the top K features from the best estimator model.

        Note:
            The model must be trained for this function to work.

        :param original_pvar_file: The original pvar corresponding to the full features
            in the X.
        :param to_csv_file: Optional file to output the top K features to.
        :return: An ordered dataframe of the top K features, ordered by most influential
            to least influential.
        """
        pvar_df = plink2_reader.pvar_reader(original_pvar_file, False)

        check_is_fitted(self.model.best_estimator_)
        pvar_mask = self.model.best_estimator_.steps[0][-1].get_support()
        best_k = pvar_mask.sum()

        top_k_df = pvar_df.loc[pvar_mask, :]
        logreg_coefs = self.model.best_estimator_.steps[-1][-1].coef_
        logreg_coefs_order = (-np.abs(logreg_coefs)).argsort().reshape(best_k)
        top_k_df = top_k_df.iloc[logreg_coefs_order]
        if to_csv_file is not None:
            top_k_df.to_csv(to_csv_file, index=False)
        return top_k_df


def main():
    data_path = pathlib.Path("data/pre-plinked")
    train_test_split = data_path.joinpath("train_test_split.npz")
    data = np.load(train_test_split)
    tks = TopKSelectorsExperiment(
        data["train_X"], data["train_y"], data.get("test_X"), data.get("test_y")
    )
    del data
    tks.train_model(completely_refit=False)

    experiment_dir = pathlib.Path(
        "data/logistic_regression_experiments/full_experiment"
    )
    tks.save_experiment(experiment_dir)
    return tks


if __name__ == "__main__":
    final_model = main()
    print("Completed!")
