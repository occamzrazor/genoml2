# Author: Berk Kapicioglu.
# This script loads the train file, applies a variety of feature selection methods to it,
# and saves the resulting feature selection objects to various files.

# NOTE(berk): Clean up, merge, or delete this file to use data from the cloud.
# Also update it in a way that saves both 1,000 and 10,000 best features for all of the
# feature selectors.  It should save the resulting feature selectors using file names that refers to
# how many features it selects.

from joblib import dump
import numpy as np
from sklearn import feature_selection
import time

RANDOM_STATE = 42
NUM_REDUCED_FEATURES_LIST = [1000, 10000, 50000]

# Load train objects.
train_file = "train.npz"
print('***Loading train objects***')
print(f'Numpy file: {train_file}')
npzfile = np.load(train_file, allow_pickle=True)
X_train = npzfile['X_train']
y_train = npzfile['y_train']


def mutual_info_true(x, y):
    return feature_selection.mutual_info_classif(x, y, discrete_features=True, random_state=RANDOM_STATE)


def mutual_info_false(x, y):
    return feature_selection.mutual_info_classif(x, y, discrete_features=False, random_state=RANDOM_STATE)


print('***Applying feature selectors***')

for num_reduced_features in NUM_REDUCED_FEATURES_LIST:
    print(f'Running chi with {num_reduced_features} features.')
    start = time.time()
    chi2_selector = feature_selection.SelectKBest(feature_selection.chi2,
                                                  k=num_reduced_features).fit(X_train, y_train)
    end = time.time()
    print(f'Chi selector ran in {end - start:.2f} seconds.')
    chi2_selector_file_name = f'feature_selector_chi2_{num_reduced_features}.joblib'
    dump(chi2_selector, chi2_selector_file_name)

#start = time.time()
#f_selector = feature_selection.SelectKBest(feature_selection.f_classif,
#                                           k=num_reduced_features).fit(X_train, y_train)
#end = time.time()
#print(f'F selector ran in {end - start:.2f} seconds.')


#start = time.time()
#mutual_info_true_selector = feature_selection.SelectKBest(mutual_info_true,
#                                                          k=num_reduced_features).fit(X_train,
#                                                                                      y_train)
#end = time.time()
#print(f'Discrete MI selector ran in {end - start:.2f} seconds.')

# NOTE(berk): Disabled both because it takes too much time and because it doesn't make sense statistically.
#start = time.time()
#mutual_info_false_selector = feature_selection.SelectKBest(mutual_info_false,
#                                                           k=num_reduced_features).fit(X_train,
#                                                                                       y_train)
#end = time.time()
#print(f'Continous MI selector ran in {end - start}')

#print('***Saving feature selector objects to files***')

#chi2_selector_file_name = 'feature_selector_' + 'chi2' + '.joblib'
#f_selector_file_name = 'feature_selector_' + 'f' + '.joblib'
#mutual_info_true_selector_file_name = 'feature_selector_' + 'mutual_info_true' + '.joblib'
#mutual_info_false_selector_file_name = 'feature_selector_' + 'mutual_info_false' + '.joblib'

#dump(chi2_selector, chi2_selector_file_name)
#dump(f_selector, f_selector_file_name)
#dump(mutual_info_true_selector, mutual_info_true_selector_file_name)
#dump(mutual_info_false_selector, mutual_info_false_selector_file_name)
