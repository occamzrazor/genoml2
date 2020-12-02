from sklearn import metrics, model_selection
from sklearn.linear_model import LogisticRegression
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier
from typing import Tuple
from sklearn.base import ClassifierMixin


class RazorLogReg:
    """
    Implements logistic regression and compares it against a ranfom classifier.
    """
    def __init__(self,
                 df: pd.DataFrame,
                 max_iter=50,
                 cv_count=5
                 ):
        X = df.drop(columns=['PD status'])
        y = df['PD status']

        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(X, y,
                                                                                                test_size=0.3,
                                                                                                random_state=42)
        self.max_iter = max_iter
        self.cv_count = cv_count

    def get_logistic_regression(self) -> ClassifierMixin:
        """
        Fits and tunes logistic regression.
        """
        #Initiallize logistic regression
        log_reg = LogisticRegression(penalty='l1',
                                       solver = 'saga',
                                       multi_class = 'multinomial')

        # Tune logistic regression with randomized search cv by maximizing the negative log-loss.
        random_search = model_selection.RandomizedSearchCV(estimator=log_reg,
                                                           param_distributions={"C": stats.randint(1, 10)},
                                                           scoring='neg_log_loss',
                                                           n_iter=self.max_iter,
                                                           cv=self.cv_count,
                                                           n_jobs=-1,
                                                           verbose=0)
        # Fit Regularized Logistic Regression
        random_search.fit(self.X_train, self.y_train)

        # Return tuned model (i.e. best estimator)
        return random_search.best_estimator_

    def get_random_classifier(self) -> ClassifierMixin:
        """
        Fit random classifier.
        """
        # Generate predictions uniformly at random with 'uniform' strategy.
        return DummyClassifier(strategy='uniform').fit(self.X_train, self.y_train)

    def predict(self, algorithm: ClassifierMixin) -> Tuple[list, list]:
        """
        Return classes and probabilities predictions for given algorithm on the test set.
        """
        return algorithm.predict(self.X_test), algorithm.predict_proba(self.X_test)

    def get_performance_table(self) -> pd.DataFrame:
        """
        Compare logistic regression's performance with the random classifier's.
        """
        # Get predictions on test set
        log_reg_predicted_classes, log_reg_predicted_probs =  self.predict(self.get_logistic_regression())
        random_predicted_classes, random_predicted_probs = self.predict(self.get_logistic_regression())

        # Construct performance table
        performance_table = pd.DataFrame({'Logistic Regression':
                                              [metrics.accuracy_score(self.y_test, log_reg_predicted_classes),
                                              metrics.precision_score(self.y_test, log_reg_predicted_classes),
                                              metrics.recall_score(self.y_test, log_reg_predicted_classes),
                                              metrics.f1_score(self.y_test, log_reg_predicted_classes),
                                              metrics.log_loss(self.y_test, log_reg_predicted_probs)],
                                          'Random Classifier':
                                              [metrics.accuracy_score(self.y_test, random_predicted_classes),
                                               metrics.precision_score(self.y_test, random_predicted_classes),
                                               metrics.recall_score(self.y_test, random_predicted_classes),
                                               metrics.f1_score(self.y_test, random_predicted_classes),
                                               metrics.log_loss(self.y_test, random_predicted_probs)]},
                                         index=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Log-Loss'])

        return performance_table