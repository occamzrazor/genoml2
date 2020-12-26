# Author: Berk Kapicioglu.
# This script loads each of the saved feature selection models,
# trains logistic regression on them, and reports their performances.

import glob
from joblib import dump, load
import numpy as np
import pickle
import re
from sklearn import feature_selection, linear_model
import time

RANDOM_STATE = 42

# Set logistic regression parameters.
REGULARIZATION = 10
CROSS_VALIDATION = 5
MAX_ITER = 1000
L1_RATIOS = [0, 0.2, 0.5, 0.8, 1]
# NOTE(berk): For debugging purposes, consider setting parameters to smaller values.
#REGULARIZATION = 2
#CROSS_VALIDATION = 3
#MAX_ITER = 100
#L1_RATIOS=[0, 0.5, 1]

# NOTE(berk): The scoring parameter values are here:
# https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules
SCORING = 'balanced_accuracy'

# NOTE(berk): If memory isn't an issue, I recommend setting it to -2.
N_JOBS = -2

# NOTE(berk): Verbosity.
LOG_REG_VERBOSITY = 0


# NOTE(berk): This definition is needed to unpickle and load objects.
def mutual_info_true(x, y):
    return feature_selection.mutual_info_classif(x, y, discrete_features=True, random_state=RANDOM_STATE)


def run_log_reg_cv(X_train, y_train):
    # Train and test logistic regression with cross-validation.
    # NOTE(berk): Turn all the function calls below into a function so that it is not repeated.
    clf = linear_model.LogisticRegressionCV(Cs=REGULARIZATION,
                                            cv=CROSS_VALIDATION,
                                            penalty='elasticnet',
                                            solver='saga',
                                            max_iter=MAX_ITER,
                                            class_weight='balanced',
                                            n_jobs=N_JOBS,
                                            verbose=LOG_REG_VERBOSITY,
                                            scoring=SCORING,
                                            l1_ratios=L1_RATIOS,
                                            random_state=RANDOM_STATE).fit(X_train, y_train)
    return clf


# Load the original train data.
train_file = "train.npz"
print('***Loading train objects***')
print(f'Numpy file: {train_file}')
npzfile = np.load(train_file, allow_pickle=True)
X_train = npzfile['X_train']
y_train = npzfile['y_train']

# Train logistic regression using all the feature selector model files.
results = {}
feature_selector_files = glob.glob('feature_selector_chi2_*.joblib')
# NOTE(berk): Update code to accommodate feature selectors with different number of reduced features.
p = re.compile('feature_selector_(chi2)_(.*).joblib')
for feature_selector_file in feature_selector_files:
    if feature_selector_file == 'feature_selector_chi2_50000.joblib':
        continue
    m = p.match(feature_selector_file)
    feature_selector_str = m.group(1)
    num_reduced_features = int(m.group(2))
    start = time.time()
    print(f'*****Processing feature selector {feature_selector_str} with {num_reduced_features} features*****')
    feature_selector_model = load(feature_selector_file)
    X_train_reduced = feature_selector_model.transform(X_train)
    clf = run_log_reg_cv(X_train_reduced, y_train)
    max_score = np.max(clf.scores_['Control'])
    results[(feature_selector_str, num_reduced_features)] = max_score
    model_file = (f'logregcv_{feature_selector_str}_{num_reduced_features}.joblib')
    dump(clf, model_file)
    end = time.time()
    print(f'*****Processed feature selector {feature_selector_str} in {end - start:.2f} seconds*****')

print('**********Final model results***********')
for (feature_selector_str, num_features), value in sorted(results.items(), key=lambda kv: kv[1], reverse=True):
    print(f'Feature selector {feature_selector_str}, number of features {num_features}: {value * 100:.2f}%')

results_file = 'model_results.pickle'
with open(results_file, 'wb') as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
