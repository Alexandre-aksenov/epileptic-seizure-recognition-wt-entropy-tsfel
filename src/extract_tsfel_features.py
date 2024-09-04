"""
Functions for extracting temporal features through the package 'tsfel'.
The main functinon is a wrapper around 'tsfel.time_series_features_extractor'.

This module brings together the functions from the script
'5_HAR_tsfel_v2_1st_si.ipynb' , then converts the result to DataFrame.

Module written on December 27th, 2023.
"""

import numpy as np
import pandas as pd
import tsfel  # 0.1.6


cfg_file = tsfel.get_features_by_domain('temporal')  # temporal features


def split0(in_str):
    """
    string "xx_yy" -> "xx"
    """
    return in_str.split('_')[0]


def decompose_arr_by_blocks(nums_cols):
    """
    Analyze the structure of a 1d-array,
    which is constant by blocks of same length:
    [x1, x1, ..., x1, x2, x2, ..., x2 , ...]

    The signal numbers in a table 'tsfel' have this format.

    Output.
        NFeat : common length of blocks.
        arr_nums_cols_inv : the array with positions of each number.
    """
    NFeat = np.count_nonzero(nums_cols == nums_cols[0])
    mat_nums_cols = nums_cols.reshape(-1, NFeat)
    # column numbers relative to each feature
    # (matrix NSignals x NFeat with constant rows).

    nums_cols_no_repetitions = mat_nums_cols[:, 0]
    arr_nums_cols_inv = np.argsort(nums_cols_no_repetitions)
    return (NFeat, arr_nums_cols_inv)


def get_tsfel_features(mat_signals, fs=None):
    """
    Main function. Extracts the time features from the signals
    placed in rows of the matrix.
    The current wrapper discards the feature names
    (keeping only the numbers of signals).
    Possible improvement: pass the feature names to the output.

    Input.
        mat_signals: raw data (ndarray NSignals x Length).
        fs (default=None raises a warning and is iterpreted by loading 'fs' from file):
            sampling rate.

    Output.
        The temporal features (DataFrame NSignals x NFeat features).
        In the first test, NFeat==14 .

    Test: see the script v4.
    """
    df_X_train = tsfel.time_series_features_extractor(cfg_file, mat_signals.T, fs=fs)
    # DataFrame with one row

    X_col_names = df_X_train.columns
    # pandas Index of strings, each one has the format "NSignal_{feature}"

    nums_cols = np.array(list(map(lambda in_s: int(split0(in_s)), X_col_names)))
    # 1d array of ints with length NSignals * NFeat
    # composed of constant blocks of same length 'NFeat'

    (NFeat, arr_nums_cols_inv) = decompose_arr_by_blocks(nums_cols)

    # convert X-train from the format 'tsfel'
    # to the indices of the original rows
    mat_X_train_tsfel_order = df_X_train.to_numpy().reshape(-1, NFeat)
    mat_X_train_orig_order = mat_X_train_tsfel_order[arr_nums_cols_inv, :]
    return pd.DataFrame(mat_X_train_orig_order)


if __name__ == '__main__':
    print(split0('1_test_feature'))
    # '1'

    ex_array = np.array([0, 0, 2, 2, 3, 3, 1, 1])
    (block_len, inv_ordering) = decompose_arr_by_blocks(ex_array)
    print("The array is composed of blocks of length", block_len)  # 2
    print("The numbers are located at positions", inv_ordering)  # [0 3 1 2]
