import joblib
import numpy as np
import pandas as pd
from sklearn import linear_model, metrics
from sklearn.dummy import DummyClassifier

DATA_PATH = './data/pre-plinked/'
features_file = DATA_PATH + 'train_test_split.npz'
data = np.load(features_file)
test_y = data['test_y']
train_X = data['train_X']
train_y = data['train_y']
test_X = data['test_X']
del data


def predict_with(filename, top_n, do_pca):
    path = 'results/'
    path_tail = 'test.npz'
    key = 'X_to_test'
    if do_pca:
        path += 'pca/'
        path_tail = 'splits.npz'
        key = 'X_test'
    model = joblib.load(path + "{}_{}_model.joblib".format(filename, top_n))
    data = np.load(path + "{}_{}_".format(filename, top_n) + path_tail)
    X_test = data[key]
    del data
    predicted_classes = model.predict(X_test)
    predicted_probabilities = model.predict_proba(X_test)
    return predicted_classes, predicted_probabilities

def get_metrics(classes):
    accuracy = metrics.accuracy_score(test_y, classes)
    precision = metrics.precision_score(test_y, classes)
    recall = metrics.recall_score(test_y, classes)
    f1 = metrics.f1_score(test_y, classes)
    balanced_accuracy = metrics.balanced_accuracy_score(test_y, classes)
    return [accuracy, precision, recall, f1, balanced_accuracy]

def get_baseline(train_X, train_y):
    most_frequent = DummyClassifier(strategy='most_frequent').fit(train_X, train_y)
    random = DummyClassifier(strategy='uniform').fit(train_X, train_y)
    del train_y
    del train_X
    most_frequent_classes = most_frequent.predict(test_X)
    random_classes = random.predict(test_X)
    mf = get_metrics(most_frequent_classes)
    r = get_metrics(random_classes)
    return mf, r
              
def all_metrics(do_pca = None):
    classes_tree_top100, probabilities_tree_top100 = predict_with('tree', 100, do_pca)
    classes_tree_top1000, probabilities_tree_top1000 = predict_with('tree', 1000, do_pca)
    classes_tree_top10000, probabilities_tree_top10000 = predict_with('tree', 10000, do_pca)

    classes_chi2_top100, probabilities_chi2_top100 = predict_with('univariate_chi2', 100, do_pca)
    classes_chi2_top1000, probabilities_chi2_top1000 = predict_with('univariate_chi2', 1000, do_pca)
    classes_chi2_top10000, probabilities_chi2_top10000 = predict_with('univariate_chi2', 10000, do_pca)

    classes_f_top100, probabilities_f_top100 = predict_with('univariate_fclassif', 100, do_pca)
    classes_f_top1000, probabilities_f_top1000 = predict_with('univariate_fclassif', 1000, do_pca)
    classes_f_top10000, probabilities_f_top10000 = predict_with('univariate_fclassif', 10000, do_pca)

    tree_100 = get_metrics(classes_tree_top100)
    tree_1000 = get_metrics(classes_tree_top1000)
    tree_10000 = get_metrics(classes_tree_top10000)

    chi2_100 = get_metrics(classes_chi2_top100)
    chi2_1000 = get_metrics(classes_chi2_top1000)
    chi2_10000 = get_metrics(classes_chi2_top10000)

    Ftest_100 = get_metrics(classes_f_top100)
    Ftest_1000 = get_metrics(classes_f_top1000)
    Ftest_10000 = get_metrics(classes_f_top10000)
    
    mf, r = get_baseline(train_X, train_y)
    return tree_100, tree_1000, tree_10000, chi2_100, chi2_1000, chi2_10000, Ftest_100, Ftest_1000, Ftest_10000, mf, r

def performance_table(do_pca = None):
    tree_100, tree_1000, tree_10000, chi2_100, chi2_1000, chi2_10000, Ftest_100, Ftest_1000, Ftest_10000, random, majority_class = all_metrics(do_pca)
    return pd.DataFrame({"Tree 100": tree_100,
                                     "Tree 1000": tree_1000,
                                     "Tree 10000": tree_10000,
                                     "Chi2 100": chi2_100,
                                     "Chi2 1000": chi2_1000,
                                     "Chi2 10000": chi2_10000,
                                     "Ftest 100": Ftest_100,
                                     "Ftest 1000": Ftest_1000,
                                     "Ftest 10000": Ftest_10000,
                                     "Random": random,
                                     "Majority Class": majority_class
                                     })


#def plot_roc_curve(probabilities):
#    fpr, tpr, thresholds = metrics.roc_curve(test_y, probabilities)
#    roc_auc_score = metrics.roc_auc_score(y_true, probabilities)
#    plt.plot(fpr, tpr, lw=2, color='blue')
#    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
#    plt.xlabel('False positive rate')
#    plt.ylabel('True positive rate')
#    plt.ylim([-0.05, 1.05])
#    plt.xlim([-0.05, 1.05])
#    plt.grid()
#    plt.title('Receiver operating characteristic AUC={0:0.2f}'.format(roc_auc_score))
