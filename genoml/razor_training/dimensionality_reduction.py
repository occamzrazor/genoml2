from sklearn import ensemble
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from typing import Tuple, Optional
from sklearn.base import ClassifierMixin


class SelectFeatures:
    """
    Implement Feature selection on high dimensional dataset.
    """

    def __init__(self,
                 n_est:int = 50
                 ):

        # Train set
        self.index = None
        self.X = None
        self.y = None
        
        # Test set
        self.index_test = None
        self.X_test = None
        self.y_test = None
        
        self.n_est = n_est
        self.features_selected = None
        
    @staticmethod
    def get_data(X, y):
        return X["participant_id"], X.drop(columns=["participant_id"]), y
        
    def get_reduced_dataset(self, X, y, model: Optional[ClassifierMixin] = None) -> pd.DataFrame:
        """
        Return dataset containing features selected in teh features_selected step
        """
        self.index, self.X, self.y = self.get_data(X, y)
        if not model: 
            model = self.features_selection()
            
        ID_df = pd.DataFrame(self.index)
        self.features_selected = model.get_support()
        X_reduced = self.X.iloc[:,self.features_selected]
        df_reduced = pd.concat([ID_df.reset_index(drop=True), 
                                self.y.reset_index(drop=True), 
                                X_reduced.reset_index(drop=True)], 
                               axis = 1, 
                               ignore_index=False)
        
        return df_reduced
    
    def get_test_set_reduced(self, X_test, y_test):
        self.index_test, self.X_test, self.y_test = self.get_data(X_test, y_test)
        ID_df_test = pd.DataFrame(self.index_test)
        X_test_reduced = self.X_test.iloc[:,self.features_selected]
        test_set_reduced = pd.concat([ID_df_test.reset_index(drop=True), 
                                self.y_test.reset_index(drop=True), 
                                X_test_reduced.reset_index(drop=True)], 
                               axis = 1, 
                               ignore_index=False)
        
        return test_set_reduced
        
    def features_selection(self) -> ClassifierMixin:
        """
        Perform feature selection using an extra-trees classifier. 
        """
        selected_features = ensemble.ExtraTreesClassifier(n_estimators=self.n_est).fit(self.X, self.y)
        
        return SelectFromModel(selected_features, prefit=True)