from sklearn import ensemble
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from typing import Tuple, Optional
from sklearn.base import ClassifierMixin


class SelectFeatures:
    """
    Implement Feature selection on high dimensional dataset.
    """

    def __init__(self,
                 n_est: int = 50,
                 method: str = 'ExtraTrees',
                 k: int = None
                 ):

        # Training set
        self.X = None
        self.y = None

        self.n_est = n_est
        self.method = method
        self.k = k
        self.features_selected = None

    @staticmethod
    def get_data(X, y):
        return X.index, X, y

    @staticmethod
    def concat_data(X_reduced, index, y):
        return pd.concat([index.reset_index(drop=True),
                          y.reset_index(drop=True),
                          X_reduced.reset_index(drop=True)],
                         axis=1,
                         ignore_index=False)

    def get_reduced_dataset(self, X, y, model: Optional[ClassifierMixin] = None) -> pd.DataFrame:
        """
        Return training dataset containing features selected in the features_selected step.
        """
        index, self.X, self.y = self.get_data(X, y)
        if not model:
            model = self.features_selection(self.method)

        ID_df = pd.DataFrame(index)
        self.features_selected = model.get_support()
        X_reduced = X.iloc[:, self.features_selected]
        return self.concat_data(X_reduced, ID_df, self.y)

    def get_test_set_reduced(self, X_test, y_test):
        """
        Return test dataset containing features selected in the features_selected step.
        """
        index, X_test, y_test = self.get_data(X_test, y_test)
        ID_df_test = pd.DataFrame(index)
        X_test_reduced = X_test.iloc[:, self.features_selected]
        return self.concat_data(X_test_reduced, ID_df_test, y_test)

    def features_selection(self, method) -> ClassifierMixin:
        """
        Perform feature selection using an extra-trees classifier. 
        """
        selected = None
        if method == 'ExtraTrees':
            selected_features = ensemble.ExtraTreesClassifier(n_estimators=self.n_est).fit(self.X, self.y)
            selected = SelectFromModel(selected_features, prefit=True)
        elif method == 'Univariate':
            if not self.k:
                raise ValueError("Have to pass number of features to be kept, k when using 'Univariate' method.")
            selected = SelectKBest(f_classif, k=self.k).fit(self.X, self.y)
        else:
            raise ValueError("Only the methods 'ExtraTrees', and 'Univariate' are supported.")
        return selected

