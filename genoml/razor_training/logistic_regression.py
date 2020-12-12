from sklearn import metrics, model_selection
from sklearn.linear_model import LogisticRegressionCV
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier
from typing import NoReturn, Optional, Tuple
from sklearn.base import ClassifierMixin
import numpy as np


class RazorLogReg:
    """
    Implement logistic regression and compares it against a random classifier.
    """

    def __init__(self,
                 train_set: Optional[pd.DataFrame] = None,
                 test_set: Optional[pd.DataFrame] = None,
                 max_iter=1000,
                 cv_count=3
                 ):
        if train_set is not None:
            self.y_train = train_set['case_control_other_latest']
            self.X_train = train_set.drop(columns=['participant_id', 'case_control_other_latest'])
        if test_set is not None:
            self.y_test = test_set['case_control_other_latest']
            self.X_test = test_set.drop(columns=['participant_id', 'case_control_other_latest'])
        
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
                                            tol=0.1
                                           )
        
        # Return fitted Regularized Logistic Regression
        return log_reg
        """
        #log_reg = LogisticRegression(penalty='elasticnet',
                                     solver='saga',
                                     multi_class='multinomial')
                                     
        #params = dict(C=stats.uniform(loc=0, scale=4), p=stats.uniform(loc=0, scale=1))
        
        #params = {'l1_penalty': [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
                  'alpha':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], 
                  'class_weight':['balanced', None]}

        # Tune logistic regression with randomized search cv by maximizing the negative log-loss.
        
        random_search = model_selection.RandomizedSearchCV(estimator=log_reg,
                                                           param_distributions=params,
                                                           scoring='neg_log_loss',
                                                           n_iter=self.max_iter,
                                                           cv=self.cv_count,
                                                           n_jobs=-1,
                                                           verbose=0)
        """
       
    def get_logistic_regression(self, log_reg: Optional[ClassifierMixin] = None):
        log_reg = log_reg
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

    def get_performance_table(self, log_reg: ClassifierMixin,
                             random_classifier: Optional[ClassifierMixin] = None, 
                             majority_classifier: Optional[ClassifierMixin] = None) -> pd.DataFrame:
        """
        Output given classifiers' performance table.
        """
        # Get predictions on test set
        log_reg_predicted_classes, log_reg_predicted_probs = self.predict(log_reg)

        # Construct performance table
        performance_table = pd.DataFrame({'Logistic Regression':
                                              [metrics.accuracy_score(self.y_test, log_reg_predicted_classes),
                                               metrics.precision_score(self.y_test, log_reg_predicted_classes, 
                                                                       average='weighted'),
                                               metrics.recall_score(self.y_test, log_reg_predicted_classes, 
                                                                   average='weighted'),
                                               metrics.f1_score(self.y_test, log_reg_predicted_classes, 
                                                               average='weighted')]},
                                         index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])
        
        if random_classifier:
            random_predicted_classes, random_predicted_probs = self.predict(random_classifier)
            performance_table['Random Classifier'] = [metrics.accuracy_score(self.y_test, random_predicted_classes),
                                                      metrics.precision_score(self.y_test, random_predicted_classes, 
                                                                             average='weighted'),
                                                      metrics.recall_score(self.y_test, random_predicted_classes,
                                                                          average='weighted'), 
                                                      metrics.f1_score(self.y_test, random_predicted_classes, 
                                                                       average='weighted')]
        if majority_classifier:
            majority_predicted_classes, majority_predicted_probs = self.predict(majority_classifier)
            performance_table['Majority Class Classifier'] = [metrics.accuracy_score(self.y_test, majority_predicted_classes),
                                                              metrics.precision_score(self.y_test, majority_predicted_classes, 
                                                                                      average='weighted'), 
                                                              metrics.recall_score(self.y_test, majority_predicted_classes, 
                                                                                  average='weighted'), 
                                                              metrics.f1_score(self.y_test, majority_predicted_classes, 
                                                                              average='weighted')]

        return performance_table
