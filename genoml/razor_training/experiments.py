import joblib
import numpy as np
from sklearn import linear_model
from sklearn import ensemble
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, chi2, mutual_info_classif
from typing import Optional

PATH = './experiments/'
DATA_PATH = './data/pre-plinked/'
C = 3
MAX_ITER = 1000
L1_RATIOS = [0, 0.2, 0.5, 0.8, 1]
CS = list(np.power(10.0, np.arange(-5, 5)))
SCORING = 'balanced_accuracy'
RANDOM_STATE = 42
K = 10000
features_file = DATA_PATH + 'train_test_split.npz'
data = np.load(features_file)
train_X = data['train_X']
train_y = data['train_y']
test_X = data['test_X']
TOP_N_LIST = [100, 1000, K]

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


def tree(filename=None):
    model = ensemble.ExtraTreesClassifier().fit(train_X, train_y)
    model = SelectFromModel(model, prefit=True)
    train_top_n(model, filename, is_tree='Tree')


def univariate_chi2(filename=None):
    model = SelectKBest(chi2, k=K).fit(train_X, train_y)
    train_top_n(model, filename)


def univariate_f_classif(filename=None):
    model = SelectKBest(f_classif, k=K).fit(train_X, train_y)
    train_top_n(model, filename)


def univariate_mutual(filename=None):
    model = SelectKBest(mutual_info_classif, k=K).fit(train_X, train_y)
    train_top_n(model, filename)


def train_top_n(model, filename, is_tree=None):
    for top_n in TOP_N_LIST:
        sorted_by_score = get_dictionary(model, top_n, is_tree)
        get_save_model(sorted_by_score, top_n, filename)


def get_dictionary(model, top_n, is_tree=None):
    boolean = model.get_support()
    feature_importance = model.scores_
    if is_tree:
        feature_importance = model.feature_importances_
    features_scores = dict(zip(np.where(boolean)[0], feature_importance[boolean]))
    sorted_by_score = dict(sorted(features_scores.items(), key=lambda item: item[1]))
    return sorted_by_score


def get_save_model(sorted_by_score, top_n, filename):
    X_to_train = get_top_features(sorted_by_score, top_n, filename)
    trained_model = fit_tune_log_reg(X_to_train, train_y)
    if filename:
        joblib.dump(trained_model, PATH + "{}_{}_model.joblib".format(filename, top_n))


def get_top_features(sorted_by_score, top_n, filename: Optional[str] = None):
    features_to_train = list(sorted_by_score.keys())[:top_n]
    X_to_train = train_X[:, features_to_train]
    if filename:
        X_to_test = test_X[features_to_train]
        np.savez("{}_{}_test.npz".format(filename, top_n),
                features=np.array(features_to_train),
                X_to_test=X_to_test)

    return X_to_train


def PCA(self, X_reduced: np.array, y: np.array) -> np.array:
    """
        Perform PCA on reduced dataset
    """
    pca = PCA(n_components=0.95)
    return pca.fit_transform(X_reduced, y)


def main():
    tree('tree')
    univariate_chi2('univariate_chi2')
    univariate_f_classif('univariate_fclassif')
    univariate_mutual('univariate_mutual')


if __name__ == "__main__":
    main()
