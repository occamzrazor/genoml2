import joblib
import numpy as np
from sklearn import linear_model
from sklearn.decomposition import PCA


DATA_PATH = './data/pre-plinked/'
RESULTS_PATH = './results/'
C = 3
MAX_ITER = 1000
L1_RATIOS = [0, 0.2, 0.5, 0.8, 1]
CS = list(np.power(10.0, np.arange(-5, 5)))
SCORING = 'balanced_accuracy'
RANDOM_STATE = 42
TOP_N_LIST = [100, 1000, 10000]
features_file = DATA_PATH + 'train_test_split.npz'
data = np.load(features_file)
train_X = data['train_X']
N = len(train_X)
train_y = data['train_y']
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

def get_dataset(train_X, filename, top_n):
    data_result = np.load(RESULTS_PATH + "{}/{}_{}_test.npz".format(filename, filename, top_n))
    selected_features = data_result['features']
    test_X = data_result['X_to_test']
    train_X_reduced = train_X[:, selected_features]
    X = np.concatenate((train_X_reduced, test_X), axis=0)
    del train_X
    del test_X
    return X


def fit_PCA(X, y):
    """
        Perform PCA on reduced dataset
    """
    pca = PCA(n_components=0.95, svd_solver='full')
    return pca.fit_transform(X, y)

def save_model(filename):
    for top_n in TOP_N_LIST:
        X = get_dataset(train_X, filename, top_n)
        X_pca = fit_PCA(X, train_y)
        pca_train_X = split_data(X_pca, filename, top_n)
        model = fit_tune_log_reg(pca_train_X, train_y)
        joblib.dump(model, "pca/{}_{}_pca.joblib".format(filename, top_n))


def split_data(X, filename, top_n):
    X_train = X[:N, :]
    X_test = X[N:len(X), :]
    np.savez("pca/{}/{}_{}.npz".format(filename, filename, top_n), X_train=X_train, X_test=X_test)
    return X_train


def main():
    save_model('tree')
    save_model('univariate_chi2')
    save_model('univariate_fclassif')

if __name__ == "__main__":
    main()

