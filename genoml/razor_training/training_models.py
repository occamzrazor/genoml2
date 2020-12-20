import joblib
import numpy as np
import dimensionality_reduction
from sklearn import linear_model

MODEL_PATH = './models/'
DATA_PATH = './data/pre-plinked/'
C = 3
MAX_ITER = 1000
L1_RATIOS = [0, 0.2, 0.5, 0.8, 1]
CS = list(np.power(10.0, np.arange(-5, 5)))
SCORING = 'balanced_accuracy'
RANDOM_STATE = 42
features_file = DATA_PATH + 'train_test_split.npz'
data = np.load(features_file)
train_X = data['train_X']
train_y = data['train_y']
test_X = data['test_X']
del data


def fit_tune_log_reg(X, y):
    model = linear_model.LogisticRegressionCV(Cs=CS,
                                              penalty='elasticnet',
                                              solver='saga',
                                              max_iter=MAX_ITER,
                                              class_weight='balanced',
                                              n_jobs=-2,
                                              verbose=0,
                                              scoring=SCORING,
                                              l1_ratios=L1_RATIOS,
                                              random_state=RANDOM_STATE,
                                              cv=C
                                             )
    return model.fit(X, y)


def select_features(filename, k=100, method=None, test=None):
    if method == 'univariate':
        if test == 'chi2':
            select_data = dimensionality_reduction.SelectFeatures(k=k, method=method, test=test)
        elif test == 'mutual':
            select_data = dimensionality_reduction.SelectFeatures(k=k, method=method, test=test)
        else:
            select_data = dimensionality_reduction.SelectFeatures(k=k, method=method)
    else:
        select_data = dimensionality_reduction.SelectFeatures(k=k)
    train_X_reduced = select_data.get_reduced_dataset(train_X, train_y)
    test_X_reduced = select_data.get_test_set_reduced(test_X)
    trained_model = fit_tune_log_reg(train_X_reduced, train_y)
    np.savez(MODEL_PATH + filename + 'split.npz',
             train_X_reduced=train_X_reduced,
             test_X_reduced=test_X_reduced)
    joblib.dump(trained_model, MODEL_PATH + filename + '.joblib')


def main():
    select_features('tree', k=1000)
    select_features('univariate_ftest', k=1000, method='Univariate')
    select_features('univariate_chi2', k=1000, method='Univariate', test='chi2')
    select_features('univariate_mutual', k=1000, method='Univariate', test='mutual')


if __name__ == "__main__":
    main()
