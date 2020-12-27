# Author: Berk Kapicioglu.
# This script constructs a pipeline that chains together
# feature selection and logistic regression.  It then runs grid search
# over it using cross-validation and saves the results.
# Code is inspired by:
# https://scikit-learn.org/stable/auto_examples/compose/plot_compare_reduction.html#sphx-glr-auto-examples-compose-plot-compare-reduction-py

from joblib import dump
import numpy as np
from sklearn import feature_selection, linear_model, model_selection, pipeline

RANDOM_STATE = 42

# Number of cross-validations.
CV=5

# Feature selection parameters.
NUM_REDUCED_FEATURES_LIST = [250, 500, 1000]

# Logistic regression parameters.
C_OPTIONS = [1e-5, 1e-4, 1e-3, 1e-2]
L1_RATIOS = [0, 0.1, 0.2]
MAX_ITER = 5000
# NOTE(berk): For debugging purposes, consider setting parameters to smaller values.
#C_OPTIONS = [1]
#L1_RATIOS=[0.5]
#MAX_ITER = 100

# NOTE(berk): The scoring parameter values are here:
# https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules
SCORING = 'balanced_accuracy'

# NOTE(berk): If memory isn't an issue, I recommend setting it to -2.
N_JOBS = 2

# NOTE(berk): Verbosity.
GENERAL_VERBOSITY = 1
LOG_REG_VERBOSITY = 0

# Grid search output file.
grid_search_output = 'grid_search_3.joblib'

# Load the original train data.
train_file = "train.npz"
print('***Loading train objects***')
print(f'Numpy file: {train_file}')
npzfile = np.load(train_file, allow_pickle=True)
X_train = npzfile['X_train']
y_train = npzfile['y_train']

# Setup grid search to conduct feature selection and logistic regression
# within cross-validation.
cachedir = 'temp_models_3'
pipe = pipeline.Pipeline([
    ('reduce_dim', feature_selection.SelectKBest(feature_selection.chi2)),
    ('classify', linear_model.LogisticRegression(penalty='elasticnet',
                                                 class_weight='balanced',
                                                 random_state=RANDOM_STATE,
                                                 solver='saga',
                                                 max_iter=MAX_ITER,
                                                 verbose=LOG_REG_VERBOSITY,
                                                 warm_start=False,
                                                 n_jobs=N_JOBS)),
], memory=cachedir, verbose=True)

param_grid = [
    {
        'reduce_dim__k': NUM_REDUCED_FEATURES_LIST,
        'classify__C': C_OPTIONS,
        'classify__l1_ratio': L1_RATIOS
    }
]

grid = model_selection.GridSearchCV(pipe,
                                    n_jobs=N_JOBS,
                                    param_grid=param_grid,
                                    scoring=SCORING,
                                    verbose=GENERAL_VERBOSITY,
                                    cv=CV)
grid.fit(X_train, y_train)

# Save the grid search model.
dump(grid, grid_search_output)
