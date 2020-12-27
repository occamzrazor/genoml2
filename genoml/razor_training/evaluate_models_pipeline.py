# Author: Berk Kapicioglu.
# This script loads each of the saved logistic regression models
# and the corresponding feature selection models and evaluates them
# on test data.

import glob
from joblib import load
import numpy as np
import re
from sklearn import dummy, feature_selection, metrics

RANDOM_STATE = 42

# Load the original train data.
train_file = "train.npz"
print('***Loading train objects***')
print(f'Numpy file: {train_file}')
npzfile = np.load(train_file, allow_pickle=True)
X_train = npzfile['X_train']
y_train = npzfile['y_train']

# Load the original test data.
test_file = "test.npz"
print('***Loading test objects***')
print(f'Numpy file: {test_file}')
npzfile = np.load(test_file, allow_pickle=True)
X_test = npzfile['X_test']
y_test = npzfile['y_test']

strategy_list = ["most_frequent", "stratified", "prior", "uniform"]

dummy_scores = {}
for strategy in strategy_list:
    clf = dummy.DummyClassifier(strategy=strategy, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = metrics.balanced_accuracy_score(y_test, y_pred)
    dummy_scores[strategy] = score

# Load the grid search results.
grid_search_cv = load('grid_search_2.joblib')

sorted_indices = np.argsort(grid_search_cv.cv_results_['rank_test_score'])
for i in sorted_indices:
    print(f"Score: {grid_search_cv.cv_results_['mean_test_score'][i] * 100:.2f}%")
    print(grid_search_cv.cv_results_['params'][i])

print('**********Final dummy results***********')
for strategy, value in sorted(dummy_scores.items(), key=lambda kv: kv[1], reverse=True):
    print(f'Strategy {strategy}: {value * 100:.2f}%')

print('**********Final model results***********')
y_pred = grid_search_cv.best_estimator_.predict(X_test)
score = metrics.balanced_accuracy_score(y_test, y_pred)
print(f'Best model: {score * 100:.2f}%')
