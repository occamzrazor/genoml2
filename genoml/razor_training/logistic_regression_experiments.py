import pathlib
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import tqdm
from sklearn import dummy, feature_selection, linear_model, metrics, model_selection
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

RANDOM_STATE = 42


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

        self.results: Optional[pd.DataFrame] = None  ## ??????????

        self.cv_count = 3
        self.max_iter = 1000
        self.l1_ratios = [0, 0.2, 0.5, 0.8, 1]
        self.Cs = list(np.power(10.0, np.arange(-5, 5)))
        self.scoring = "balanced_accuracy"
        self.model = linear_model.LogisticRegressionCV()
        self.init_model()

    def init_model(self):
        sss = model_selection.StratifiedShuffleSplit(n_splits=self.cv_count)
        self.model = linear_model.LogisticRegressionCV(
            Cs=self.Cs,
            penalty="elasticnet",
            solver="saga",
            max_iter=self.max_iter,
            class_weight="balanced",
            n_jobs=-2,
            verbose=0,
            scoring=self.scoring,
            l1_ratios=self.l1_ratios,
            random_state=RANDOM_STATE,
            cv=sss,
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
        except NotFittedError as e:
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
        dummy_freq = dummy.DummyClassifier(
            strategy="most_frequent", random_state=RANDOM_STATE
        )
        dummy_freq.fit(self.train_x, self.train_y)
        freq_score = _score_model(dummy_freq, self.test_x, self.test_y)

        dummy_strat = dummy.DummyClassifier(
            strategy="stratified", random_state=RANDOM_STATE
        )
        dummy_strat.fit(self.train_x, self.train_y)
        strat_score = _score_model(dummy_strat, self.test_x, self.test_y)

        return {"frequency_dummy": freq_score, "stratified_score": strat_score}


def _score_model(clf, test_x, test_y) -> Dict[str, float]:
    score = metrics.balanced_accuracy_score(test_y, clf.predict(test_x))
    return {"balanced_accuracy": score}


class TopKSelectorsExperiment(object):
    def __init__(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        test_X: Optional = None,
        test_y: Optional = None,
        ks: Optional[List[int]] = None,
    ):
        self.train_x = train_X
        self.train_y = train_y

        self.test_x = test_X
        self.test_y = test_y

        if ks is None:
            ks = [100, 1000, 10000]
        ks = [min(self.train_x.shape[1], k) for k in ks]
        self.ks = list(set(ks))

        self._logreg_experiments: Dict[int, LogRegExperiment] = dict()

        self.selector_model = feature_selection.SelectKBest(
            feature_selection.chi2, k=max(self.ks)
        )
        self.feature_importance = np.empty(0, dtype=np.int)

    @classmethod
    def load_experiment(
        cls, data_path, experiment_dir=None, ks=None
    ) -> "TopKSelectorsExperiment":
        data_path = pathlib.Path(data_path)
        data = np.load(data_path.joinpath("train_test_split.npz"))

        self = cls(
            data["train_X"],
            data["train_y"],
            data.get("test_X"),
            data.get("test_y"),
            ks=ks,
        )
        if not experiment_dir:
            return self

        experiment_dir = pathlib.Path(experiment_dir)
        selector_model_file = experiment_dir.joinpath("selector_model.joblib")
        if selector_model_file.exists():
            with open(selector_model_file, "rb") as fo:
                self.model = joblib.load(fo)

        for logreg_exp_dir in experiment_dir.glob("selected_*"):
            k = logreg_exp_dir.name.split("_")[-1]
            logreg_exp = LogRegExperiment.load_experiment(logreg_exp_dir)
            self._logreg_experiments[k] = logreg_exp

        return self

    def save(self, directory):
        directory = pathlib.Path(directory)
        directory.mkdir(exist_ok=True)

        if self.selector_model:
            with open(directory.joinpath("selector_model.joblib"), "wb") as fi:
                joblib.dump(self.selector_model, fi)

        for k, exp in self._logreg_experiments.items():
            k_selected_dir = directory.joinpath(f"selected_{k}")
            exp.save_experiment(k_selected_dir)

    def fit_feature_selection_model(self, scoring_funct=None) -> np.ndarray:
        if scoring_funct is None:
            scoring_funct = feature_selection.chi2
        self.selector_model = feature_selection.SelectKBest(
            scoring_funct, k=max(self.ks)
        ).fit(self.train_x, self.train_y)
        self.feature_importance = (-self.selector_model.scores_).argsort()
        return self.feature_importance

    def initialize_logreg_experiments(self):
        for k in self.ks:
            # Initialize the logreg experiments
            variant_mask = self.feature_importance < k
            logreg_exp = LogRegExperiment(self.train_x[:, variant_mask], self.train_y)
            if self.test_x is not None:
                logreg_exp.test_x = self.test_x[:, variant_mask]
                logreg_exp.test_y = self.test_y

            self._logreg_experiments[k] = logreg_exp

    def fit_logreg_experiments(self, *args, **kwargs):
        if not self._logreg_experiments:
            self.initialize_logreg_experiments()
        pbar = tqdm.tqdm(
            self._logreg_experiments.items(), desc="Training the LogReg experiments"
        )
        for k, exp in pbar:
            pbar.set_description(
                f"Training the LogReg experiments; k={k}", refresh=True
            )
            exp.train_model(*args, **kwargs)


def main():
    data_path = pathlib.Path("data/pre-plinked")
    experiment_dir = pathlib.Path("data/logistic_regression_experiments")
    if experiment_dir.joinpath("selector_model.joblib").exists():
        tks = TopKSelectorsExperiment.load_experiment(
            data_path, experiment_dir=experiment_dir
        )
    else:
        tks = TopKSelectorsExperiment.load_experiment(data_path)
        tks.fit_feature_selection_model()

    tks.fit_logreg_experiments(refit=False)
    tks.save(experiment_dir)

    for exp in tks._logreg_experiments.values():
        print(exp.results)


if __name__ == "__main__":
    main()
    completed = True
    print("Completed!")
