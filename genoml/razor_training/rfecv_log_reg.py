import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
import joblib

SAVE_PATH = 'rfecv_log_reg.joblib'
OUTPUT_PATH = '/Users/mdmcastanos/genomlRazor/genoml/razor_training/data_pre-plinked/'
C = 3
MAX_ITER = 1000
L1_RATIO = 0.5
STEP = 0.2
SCORING = 'balanced_accuracy'

train_X, test_X, train_y, test_y = train_test_split.get_train_test(OUTPUT_PATH + "ldpruned_data.pgen",
                                                                   OUTPUT_PATH + "ldpruned_data.psam",
                                                                   OUTPUT_PATH + "amp_pd_case_control.tsv",
                                                                   output_file=OUTPUT_PATH + "train_test_split.npz"
                                                                   )

del test_X
del test_y

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
    main('multinomial')
