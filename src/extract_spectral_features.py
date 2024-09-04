"""
Functions for extracting spectral features.
These functions are used in Homework 3 of the class ML_Advanced.

Test of spectral functions: see the script 'spectral_feat_1_signal.ipynb'.
"""

import numpy as np
from scipy.signal import welch, find_peaks


def welch64(in_sig):
    """
    Welch spectral estimate with window length of length 64
    (adapted to the signals being of length 178).

    Args:
        in_sig (1d Series): the signal.

    Returns:
        frequencies, PSD (1d arrays).
    """
    return welch(in_sig, nperseg=64)


def PSD_welch64(in_sig):
    """
    Equivalent to 'welch64',
    but returns only the PSD of a signal.
    Useful for running in loop.
    """
    _, PSD = welch64(in_sig)
    return PSD


def get_first_n(x, no_peaks=4):
    """
    First 'no_peaks' values of a tuple or 1d ndarray.

    If a signal is shorter than 'no_peaks',
    it is padded by zeroes to the length 'no_peaks'.

    Args:
        x (1d ndarray): 1st full sequence
        no_peaks (int, optional): nb of first values to extract. Defaults to 4.

    Returns:
        starting values of each sequence: tuple of 2 lists.
    """
    x_ = list(x)
    if len(x_) >= no_peaks:
        return x_[:no_peaks]
    else:
        missing_no_peaks = no_peaks - len(x_)
        return x_ + [0] * missing_no_peaks


def get_first_n_peaks(x, no_peaks=4):
    """
    Sorts a signal, then extracts indices and values of the first 'no_peaks' local maxima. 
    If the number of peaks is smaller than 'no_peaks', the last peaks and values are set to zero.

    Args:
        x (1d ndarray): the signal
        no_peaks (int, optional): nb of first peaks to extract. Defaults to 4.

    Returns:
        first indices then first values (1d array of length 2*no_peaks).
    """
    peaks = find_peaks(x)[0]
    x_sorted = np.sort(peaks)
    first_ind = get_first_n(x_sorted, no_peaks=no_peaks)
    first_peak_vals = get_first_n(x[x_sorted])
    return np.array(first_ind + list(first_peak_vals))


def first_n_spectral_peaks(in_sig, no_peaks=4):
    return get_first_n_peaks(PSD_welch64(in_sig), no_peaks=no_peaks)


# testing examples
if __name__ == '__main__':
    # the 1st three results are identical.
    test_3_peaks = np.array([0., 2., 0., 1., 0., 3., 0.])
    print(get_first_n_peaks(test_3_peaks))
    # [1. ,3. ,5. ,0. , 2. ,1. ,3. , 0.]

    nonzero_at_zero = np.array([1., 2., 0., 1., 0., 3., 0.])
    print(get_first_n_peaks(nonzero_at_zero))
    # [1. ,3. ,5. ,0. , 2. ,1. ,3. , 0.] # OK

    ex_peak_at_border = np.array([0., 2., 0., 1., 0., 3., 0., 0., 4.])
    print(get_first_n_peaks(ex_peak_at_border))
    # [1., 3. , 5., 0., 2., 1., 3., 0.]
    # the function 'find_peaks' does not detect peaks at borders.

    ex_4_peaks = np.array([0., 2., 0., 1., 0., 3., 0., 0., 4., 0.])
    print(get_first_n_peaks(ex_4_peaks))
    # [1., 3. , 5., 8., 2., 1., 3., 4.]
