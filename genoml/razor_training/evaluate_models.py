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


# NOTE(berk): This function is needed to unpickle and load objects.
def mutual_info_true(x, y):
    return feature_selection.mutual_info_classif(x, y, discrete_features=True, random_state=RANDOM_STATE)


logreg_test_scores = {}
logreg_cv_scores = {}
p = re.compile('logregcv_(.*)_(.*).joblib')
logreg_files = glob.glob('logregcv_*_*.joblib')
for logreg_file in logreg_files:
    m = p.match(logreg_file)
    feature_selector_str = m.group(1)
    num_reduced_features = int(m.group(2))
    clf = load(logreg_file)
    max_cv_score = np.max(clf.scores_['Control'])
    logreg_cv_scores[(feature_selector_str, num_reduced_features)] = max_cv_score
    # Load the feature selection model.
    feature_selector_file = f'feature_selector_{feature_selector_str}_{num_reduced_features}.joblib'
    feature_selector_model = load(feature_selector_file)
    X_test_reduced = feature_selector_model.transform(X_test)
    y_pred = clf.predict(X_test_reduced)
    score = metrics.balanced_accuracy_score(y_test, y_pred)
    logreg_test_scores[(feature_selector_str, num_reduced_features)] = score

print('**********Final dummy results***********')
for strategy, value in sorted(dummy_scores.items(), key=lambda kv: kv[1], reverse=True):
    print(f'Strategy {strategy}: {value * 100:.2f}%')

print('**********Final model results***********')
for (feature_selector_str, num_features), value in sorted(logreg_test_scores.items(), key=lambda kv: kv[1], reverse=True):
    cv_score = logreg_cv_scores[(feature_selector_str, num_features)]
    print(f'Feature selector {feature_selector_str}, number of features {num_features} test score: {value * 100:.2f}%')
    print(f'Feature selector {feature_selector_str}, number of features {num_features} CV score: {cv_score * 100:.2f}%')
