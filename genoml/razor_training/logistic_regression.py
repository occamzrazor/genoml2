from sklearn import metrics, model_selection
from sklearn.linear_model import LogisticRegressionCV
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier
from typing import Optional, Tuple
from sklearn.base import ClassifierMixin
import numpy as np



class RazorLogReg:
    """
    Implement logistic regression, random and majority class classifiers.
    """

    def __init__(self,
                 train_set: Optional[pd.DataFrame] = None,
                 test_set: Optional[pd.DataFrame] = None,
                 max_iter=5000,
                 cv_count=3
                 ):
        if train_set is not None:
            self.y_train = train_set['case_control_other_latest']
            self.X_train = train_set.drop(columns=['case_control_other_latest'])
        if test_set is not None:
            self.y_test = test_set['case_control_other_latest']
            self.X_test = test_set.drop(columns=['case_control_other_latest'])
        
        self.max_iter = max_iter
        self.cv_count = cv_count

    def logistic_regression(self) -> ClassifierMixin:
        """
        Fits and tunes logistic regression.
        """
        # Initiallize logistic regression (tol is set to a high value to finish with the example faster)
        log_reg = LogisticRegressionCV(penalty='elasticnet',
                                       solver='saga',
                                       multi_class='multinomial', 
                                       scoring='neg_log_loss',
                                       Cs=list(np.power(10.0, np.arange(-10, 10))),
                                       l1_ratios=[0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
                                       class_weight='balanced', 
                                       max_iter=self.max_iter, 
                                       cv=self.cv_count,
                                       n_jobs=-1
                                      )
        
        # Return fitted Regularized Logistic Regression
        return log_reg
       
    def get_logistic_regression(self, log_reg: Optional[ClassifierMixin] = None):
        if not log_reg:
            log_reg = self.logistic_regression()
        return log_reg.fit(self.X_train, self.y_train)
            
    def get_random_classifier(self) -> ClassifierMixin:
        """
        Fit random classifier.
        """
        # Generate predictions uniformly at random with 'uniform' strategy.
        return DummyClassifier(strategy='uniform').fit(self.X_train, self.y_train)
    
    def get_majority_class_classifier(self) -> ClassifierMixin:
        """
        Fit majority-class classifier.
        """
        # Generate predictions based on the most frequent class.
        return DummyClassifier(strategy='most_frequent').fit(self.X_train, self.y_train)

    def predict(self, algorithm: ClassifierMixin) -> Tuple[list, list]:
        """
        Return classes and probabilities predictions for given algorithm on the test set.
        """
        return algorithm.predict(self.X_test), algorithm.predict_proba(self.X_test)
    
    @staticmethod
    def get_metrics(self, y_test, predictions:list, avg:str) -> list:
        accuracy = metrics.accuracy_score(y_test, predictions)
        precision = metrics.precision_score(y_test, predictions, average=avg, zero_division=1)
        recall = metrics.recall_score(y_test, predictions, average=avg)
        f1 = metrics.f1_score(y_test, predictions, average=avg)
        return [accuracy, precision, recall, f1]
