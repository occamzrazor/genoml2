# Author: Berk Kapicioglu.
# This is the script I use to process the data and train on it.
# Ideally, we would already have the train and test splits stored somewhere in the cloud
# so that we can all use those splits to run our experiments and compare results.

from joblib import dump
import numpy as np
import pandas as pd
from sklearn import feature_selection, linear_model, model_selection

RANDOM_STATE = 42

# Feature file.
features_file = "plink_numpy.npy"
# NOTE(berk): For debugging purposes, a small numpy array.
#features_file = "plink_small.npy"
# Label file.
labels_file = "latest_labels.tsv"

X = np.load(features_file)
df_y = pd.read_csv(labels_file, sep='\t')

# Collapse 3 classes into 2 classes.
df_y.loc[df_y.case_control_other_latest == 'Other', 'case_control_other_latest'] = 'Case'

# NOTE(berk): I manually flip 0 and 2 so that 0 is the most frequent entry.
X[X == 2] = -1
X[X == 0] = -2
X[X == -1] = 0
X[X == -2] = 2

# NOTE(berk): -9 values should be replaced by median/mode imputation for each column here.
# For simplicity, I replace them with 0.
X[X == -9] = 0

# Sanity check the frequencies of values in features matrix.
# By this point:
# * 0 and 2 should be flipped and 0 should be the most frequent entry.
# * '-9' depicts missing values and should be replaced by a median imputation for each column.
# NOTE(berk): Temporarily disabled.
#value_frequencies = pd.value_counts(X.ravel(), normalize=True, dropna=False)
#print('***Printing the frequency of each entry in the features matrix***')
#print(value_frequencies)

y = df_y['case_control_other_latest']
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1,
                                                                    random_state=RANDOM_STATE)

# Remove reference to original feature array in case garbage collection kicks in to save memory.
del X

with open('train_test.npy', 'wb') as f:
    np.save(f, X_train)
    np.save(f, X_test)
    np.save(f, y_train)
    np.save(f, y_test)

# Set learning parameters.
C = 3
MAX_ITER = 1000
L1_RATIOS = [0, 0.2, 0.5, 0.8, 1]
STEP = 0.2
# NOTE(berk): For debugging purposes, consider setting parameters to smaller values.
#C = 2
#MAX_ITER = 100
#L1_RATIOS=[0, 0.5, 1]
#STEP = 0.4

# NOTE(berk): The scoring parameter values are here:
# https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules
SCORING = 'balanced_accuracy'

# Train and test logistic regression with cross-validation.
# NOTE(berk): Turn all the function calls below into a function so that it is not repeated.
clf = linear_model.LogisticRegressionCV(Cs=C,
                                        penalty='elasticnet',
                                        solver='saga',
                                        max_iter=MAX_ITER,
                                        class_weight='balanced',
                                        n_jobs=-2,
                                        verbose=0,
                                        scoring=SCORING,
                                        l1_ratios=L1_RATIOS,
                                        random_state=RANDOM_STATE)
clf_rf = feature_selection.RFECV(clf, step=STEP, cv=C, scoring=SCORING, n_jobs=-2, verbose=1).fit(X_train, y_train)

# Save the model to file.
model_file_name = 'rfecv_' + SCORING + '.joblib'
dump(clf_rf, model_file_name)
