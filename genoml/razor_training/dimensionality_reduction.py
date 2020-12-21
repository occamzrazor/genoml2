from sklearn import ensemble
import pandas as pd
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, chi2, mutual_info_classif
from typing import Optional, NoReturn
from sklearn.base import ClassifierMixin
import numpy as np
from sklearn.decomposition import PCA


class SelectFeatures:
    """
    Implement various feature selection methods.
    """

    def __init__(self,
                 k: int = 10000,
                 method: str = 'ExtraTrees',
                 test: str = None, 
                 outfile: str = None,
                 run_PCA: bool = False
                 ):

        self.method = method
        self.k = k
        self.test = test
        self.features_selected = None
        self.outfile = outfile

    def get_reduced_dataset(self, X: np.array, y: np.array, model: Optional[ClassifierMixin] = None) -> np.array:
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
        X_reduced = model.transform(X)
        if run_PCA: 
            # Apply PCA on sellected features
            X_reduced = self.PCA(X_reduced, y)
        return X_reduced 

    def get_test_set_reduced(self, X_test: np.array):
        """
        Return test dataset containing features selected in the features_selected step.
        """
        X_test_reduced = X_test[:, self.features_selected]
        return X_test_reduced

    def features_selection(self, X: np.array, y: np.array) -> ClassifierMixin:
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
    
    def PCA(self, X_reduced: np.array, y: np.array) -> np.array:
        """
        Perform PCA on reduced dataset
        """
        pca = PCA(n_components=0.95)
        return pca.fit_transform(X_reduced, y) 
        
