# Author: Berk Kapicioglu.
# This script loads the plink files, processes them, and
# saves them as train and test files in compressed numpy format.

# NOTE(berk): Consider removing or merging this code such that
# the train and test files, which are compressed and npz, are available in the cloud,
# but based on similar processing to what is done below.

import numpy as np
import pandas as pd
from sklearn import model_selection

RANDOM_STATE = 42

# Feature file.
features_file = "plink_numpy.npy"
# NOTE(berk): For debugging purposes, a small numpy array.
#features_file = "plink_small.npy"
# Label file.
labels_file = "latest_labels.tsv"
# Output files.
train_file = "train.npz"
test_file = "test.npz"

print('***Processing the following input files***')
print(f'Plink file: {features_file}')
print(f'Labels file: {labels_file}')

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
value_frequencies = pd.value_counts(X.ravel(), normalize=True, dropna=False)
print('***Printing the frequency of each entry in the features matrix***')
print(value_frequencies)

y = df_y['case_control_other_latest']
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1,
                                                                    random_state=RANDOM_STATE)

np.savez_compressed(train_file, X_train=X_train, y_train=y_train)
np.savez_compressed(test_file, X_test=X_test, y_test=y_test)

print('***Saved train and test files***')
print(f'Train file: {train_file}')
print(f'Test file: {test_file}')
