# Author: Berk Kapicioglu.
# This is the script I load the pre-trained models and pre-prepared test splits
# to evaluate them against other dummy classifiers.

from joblib import load
import numpy as np
from sklearn import dummy, metrics

RANDOM_STATE = 42

with open('train_test.npy', 'rb') as f:
    X_train = np.load(f, allow_pickle=True)
    X_test = np.load(f, allow_pickle=True)
    y_train = np.load(f, allow_pickle=True)
    y_test = np.load(f, allow_pickle=True)

model_file_name = 'rfecv_balanced_accuracy.joblib'
clf_rf = load(model_file_name)

clf1 = dummy.DummyClassifier(strategy='most_frequent', random_state=RANDOM_STATE)
clf1.fit(X_train, y_train)
metrics.balanced_accuracy_score(y_test, clf1.predict(X_test))

clf2 = dummy.DummyClassifier(strategy='stratified', random_state=RANDOM_STATE)
clf2.fit(X_train, y_train)
metrics.balanced_accuracy_score(y_test, clf2.predict(X_test))
