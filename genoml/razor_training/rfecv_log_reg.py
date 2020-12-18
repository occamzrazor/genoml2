from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
import joblib
import numpy as np

SAVE_PATH = '/Users/mdmcastanos/genomlRazor/genoml/razor_training/models/rfecv_log_reg.joblib'
DATA_PATH = '/Users/mdmcastanos/genomlRazor/genoml/razor_training/data_pre-plinked/'
C = 3
MAX_ITER = 1000
L1_RATIO = 0.5
STEP = 0.2
SCORING = 'balanced_accuracy'
CLASS = 'multinomial'
features_file = DATA_PATH + 'train_test_split.npz'
data = np.load(features_file)
train_X = data['train_X']
train_y = data['train_y']
del data    

def main(multi_labels):
    log_reg_object = LogisticRegression(penalty='elasticnet',
                                        solver='saga',
                                        multi_class=multi_labels,
                                        l1_ratio=L1_RATIO,
                                        class_weight='balanced',
                                        max_iter=MAX_ITER,
                                        n_jobs=-2,
                                        verbose=0
                                        )

    selector = RFECV(log_reg_object, step=STEP, cv=C, scoring=SCORING, verbose=1)
    trained_classifier = selector.fit(train_X, train_y)
    joblib.dump(trained_classifier, SAVE_PATH)


if __name__ == "__main__":
    main(CLASS)
