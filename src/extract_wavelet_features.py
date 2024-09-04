"""
The functions relative to feature extraction using wavelets
for the Homework number 3.

The differential entropy
is used as a measure of the amplitude of a signal.
"""

import numpy as np
from scipy.stats import differential_entropy
from tqdm.autonotebook import tqdm
import pywt


def calculate_statistics(list_values):
    """
    stat series -> 9 statistical quantities
    """
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values**2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]


def calculate_crossings(list_values):
    """
    stat series (list or 1d-array) ->
    nb of zero crossings, mean crossings
        (list of 2 ints >=0)

    Is applied to wavelet transorm in functions below.
    See this article for explanation:
    https://www.di.ens.fr/~mallat/papiers/MallatZero91.pdf
    """
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(
        np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]


def get_features(list_values):
    """
    stat series (1d array) -> all 12 previous statistics together
    """
    entropy = differential_entropy(list_values)
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
    return [entropy] + crossings + statistics


def get_wt_features(dataset, waveletname):
    """
    Main function.
    Features extracted using wavelet transform.

    Args:
        dataset (NSamples x time x NSignals per sample).
        waveletname (str): the name of wavelet
            ('rbio3.1' in the script).

    Returns.
    X == (final) uci_har_features : the extracted features.
        matrix of shape:  NSamples x (NSignals * NW * 12)
            where 12 is the length of the result
            of the inner function 'get_features',
        NW is the number of wavelets
        (== 6 in case of signals of length 128 or 172).

    Each row of X is a flattened version of tensor
    NSignals x NW (<- pywt.wavedec) x 12 (<- get_features)

    """
    uci_har_features = []
    for signal_no in tqdm(range(0, len(dataset)), leave=False):
        features = []
        # this list will contain all features relative to a sample.

        for signal_comp in range(0, dataset.shape[2]):
            signal = dataset[signal_no, :, signal_comp]
            list_coeff = pywt.wavedec(signal, waveletname)
            # list_coeff: list of length NW of DWT signals (1d numpy arrays)
            # https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html#multilevel-decomposition-using-wavedec
            for coeff in list_coeff:
                # this small loop adds NW * 12 numbers at the end of 'features'
                # to the list of lists 'features'.
                features += get_features(coeff)
        uci_har_features.append(features)
    X = np.array(uci_har_features)
    return X
