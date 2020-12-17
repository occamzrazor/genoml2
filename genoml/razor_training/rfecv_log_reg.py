import joblib
from sklearn.feature_selection import RFECV
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

def main(save_path):
    """ Problem: Train and save logistic regression for the RFECV method.
        
        Args:
        save_path: Path to save trained model.
    """
    
    path = "/Users/mdmcastanos/genoml2/plink_files/"
    npy_file = path + "full_plink2_matrix.npy"
    tsv_file = path + "latest_labels.tsv"
    df_y = pd.read_csv(tsv_file, sep='\t')
    df = pd.DataFrame(np.load(npy_file))

    df = df.assign(participant_id=df_y['participant_id'], case_control_other_latest=df_y['case_control_other_latest'])
    df = df.set_index('participant_id')
    df.reset_index(drop=True, inplace=True)
    
    train, test = model_selection.train_test_split(df, test_size=0.3, random_state=42)
    
    del df
    del df_y
    del test
    
    log_reg_fitted = LogisticRegression(penalty='elasticnet',
                                        solver='saga',
                                        multi_class='multinomial',
                                        l1_ratio=0.5,
                                        class_weight='balanced',
                                        max_iter=1000,
                                        n_jobs=-1
                                      )
                                      
    selector = RFECV(log_reg_fitted, step=2000)
    
    y_train = train['case_control_other_latest']
    train.drop(columns=['case_control_other_latest'], inplace=True)
    trained_classifier = selector.fit(train, y_train)
    joblib.dump(trained_classifier, save_path)

if __name__ == '__main__':
    main(save_path='RFECV_log_reg')
