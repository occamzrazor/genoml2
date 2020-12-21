from sklearn import ensemble
import pandas as pd
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, chi2, mutual_info_classif
from typing import Optional
from sklearn.base import ClassifierMixin
import numpy as np


class SelectFeatures:
    """
    Implement various feature selection methods.
    """

    def __init__(self,
                 k: int = 10000,
                 method: str = 'ExtraTrees',
                 test: str = None, 
                 outfile: str = None
                 ):

        self.method = method
        self.k = k
        self.test = test
        self.features_selected = None
        self.outfile = outfile

    def get_reduced_dataset(self, X, y, model: Optional[ClassifierMixin] = None) -> pd.DataFrame:
        """
        Return training dataset containing features selected in the features_selected step.
        """
        if not model:
            model = self.features_selection(X, y)
        # Save selected features by the model
        self.features_selected = model.get_support()
        if outfile:
            np.save(self.outfile, np.where(self.features_selected))
        # Return reduced dataset
        return model.transform(X)

    def get_test_set_reduced(self, X_test):
        """
        Return test dataset containing features selected in the features_selected step.
        """
        X_test_reduced = X_test[:, self.features_selected]
        return X_test_reduced

    def features_selection(self, X, y) -> ClassifierMixin:
        """
        Perform feature selection using an extra-trees classifier. 
        """
        if self.method == 'ExtraTrees':
            selected_features = ensemble.ExtraTreesClassifier(n_estimators=self.k).fit(X, y)
            selected = SelectFromModel(selected_features, prefit=True)
        elif self.method == 'Univariate':
            if self.test == 'chi2':
                selected = SelectKBest(chi2, k=self.k).fit(X, y)
            elif self.test == 'mutual':
                selected = SelectKBest(mutual_info_classif, k=self.k).fit(X, y)
            else:
                selected = SelectKBest(f_classif, k=self.k).fit(X, y)
        else:
            raise ValueError("Only the methods 'ExtraTrees', and 'Univariate' are supported.")
        return selected
