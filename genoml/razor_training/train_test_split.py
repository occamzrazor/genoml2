"""Splits the data into train/test numpy arrays."""


import numpy as np
import pandas as pd
from sklearn import feature_selection, linear_model, model_selection

from genoml.preprocessing import plink2_reader

RANDOM_STATE = 42


def get_train_test(pgen_file, psam_file, case_control_file, output_file=None):
    patient_df = plink2_reader.psam_reader(psam_file)[["IID"]]
    patient_labels = pd.read_csv(case_control_file)[
        ["participant_id", "diagnosis_latest", "case_control_other_latest"]
    ]
    patient_labels["y"] = patient_labels["case_control_other_latest"] == "Case"

    patient_df = patient_df.merge(
        patient_labels, left_on="IID", right_on="participant_id", how="left"
    )
    y = (patient_df["case_control_other_latest"] == "Case").to_numpy()
    X = plink2_reader.pgen_reader(pgen_file, None, impute="median")

    sss = model_selection.StratifiedShuffleSplit(
        n_splits=1, test_size=0.3, random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(sss.split(X, y))

    train_X = X[train_idx]
    test_X = X[test_idx]
    train_y = y[train_idx]
    test_y = y[test_idx]
    del X

    if output_file:
        np.savez(
            "data/pre-plinked/train_test_split.npz", train_X=train_X, test_X=test_X, train_y=train_y, test_y=test_y
        )

    return train_X, test_X, train_y, test_y


if __name__ == "__main__":
    get_train_test(
        "data/pre-plinked/ldpruned_data.pgen",
        "data/pre-plinked/ldpruned_data.psam",
        "data/pre-plinked/amp_pd_case_control.csv",
        output_file="data/pre-plinked/train_test_split.npz"
    )

