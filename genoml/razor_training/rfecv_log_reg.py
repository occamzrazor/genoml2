import joblib
from logistic_regression import RazorLogReg
from sklearn.feature_selection import RFECV
import pandas as pd
from dimensionality_reduction import SelectFeatures
import numpy as np
from sklearn import model_selection

def main(save_path):
    """ Problem: Train and save logistic regression for various feature selection methods.
        
        Args:
        model: Path to CSV file containing dataset for training.
        X_train_path: Path to CSV file containing predictors's dataset for training.
        y_train_path: Path to CSV file containing dependent variable's dataset for training.
        save_path: Path to save trained model.
    """
    path = "/Users/mdmcastanos/genoml2/plink_files/"
    npy_file = path + "full_plink2_matrix.npy"
    tsv_file = path + "latest_labels.tsv"
    df_y = pd.read_csv(tsv_file, sep='\t')
    df_X = pd.DataFrame(np.load(npy_file))

    df = df_X.assign(participant_id=df_y['participant_id'], case_control_other_latest=df_y['case_control_other_latest'])
    df = df.set_index('participant_id')
    df.reset_index(drop=True, inplace=True)

    X = df.drop(columns=['case_control_other_latest'])
    y = df['case_control_other_latest']
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=42)
    
    training_set = pd.concat([X_train, y_train], axis=1)
    rfecv = RazorLogReg(train_set=training_set)
    rfecv_log_reg = rfecv.logistic_regression()
    selector = RFECV(rfecv_log_reg, step=1000, cv = 3)
    trained_classifier = selector.fit_transform(X, y)(X_train, y_train)
    joblib.dump(trained_classifier, save_path)
"""
class train_models():
    
    def __init__(self,
                 X_train,
                 y_train):
        # Import training data
        self.X = X_train
        self.y= y_train

    def RFECV(self):
        
        # Concatenate data
        training_set = pd.concat([X_train, y_train], axis=1)
        
        # Train Logistic Regression and apply RFECV
        rfecv = RazorLogReg(train_set=training_set)
        rfecv_log_reg = rfecv.logistic_regression()
        selector = RFECV(rfecv_log_reg, step=1000, cv=3)
        
        # Return refitted model
        return selector.fit(X_train, y_train)
"""
if __name__ == '__main__':
    main(save_path='RFECV_log_reg')

