# epileptic-seizure-recognition-wt-entropy-tsfel
Epilepsy detection using EEG data.

Epileptic activity can be recognized by a specialist as activity with much higher signal amplitude (see [Vila-Vidal et als]). Despite this observation, automatic classification is expected to be hindered by differences in distance between the epicenter and the measurement electrode. The models presented here were not given any *a priori* information to use the signal amplitude as feature. Despite this, the classification models perform quite well.

This repository contains three folders: "src" contains notebooks with classification models and modules for feature extraction, "res" contains pictures produced durng Exploratory Analysis and "dat" contains the raw data.

Reference:

M. Vila-Vidal, C. P. Enríquez, A. Principe, R. Rocamora, G. Deco, A. T. Campo "Low entropy map of brain oscillatory activity identifies spatially localized events: A new method for automated epilepsy focus prediction", Neuroimage 208 (2020)

<b>About the dataset.</b>

Each row contains the signal of one EEG electrode during 1 second. The dataset comes from:
https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition

It contains 11500 rows and 178 features. The response variable contains the category of the 178-dimensional input vector (5 classes, one of them being seizure activity).

<b>About the problem.</b>

The goal is to recognize an Epileptic seizure against the rest. A task of binary classification is treated in this repository.

<b>Selected models.</b>

The features are treated as a time series of 178 points. Three ways to extract features lead to different classifiers:
* use the raw data as features (notebook: <code>classification_raw_data.ipynb</code>)
* apply discrete wavelet transform, then use classical statistical quantites (percentiles, mean, entropy, number of zero crossigs etc); also estimate the spectrum (Welch estimator) and use the positions and values of first spectral peaks (notebook:  <code>classification_wavelets_fourier.ipynb</code>).
* use the features extracted by <code>tsfel.time_series_features_extractor</code> (notebook:  <code>classification_tsfel.ipynb</code>).

A Random Forest classifier (50 estimators) is trained on each set of extracted features.

<b>Results.</b>

Each model's performance is measured by its f1-score for the seizure class on test set and the number of features used for classification. The performances of all 3 models seem quite correct, which is consistent with the observation that the normal and epileptic signals tend to present different amplitudes (notebook <code>explore.ipynb</code>).

| Feature extraction | f1-score | number of features |
| ------------------ | -------- | ------------------ |
| Raw data | 0.91 | 178 |
| Fourier, wavelets | 0.96 | 80 |
| <code>tsfel</code> | 0.95 | 10 |

<b>Possible improvement.</b>

The solution can be improved by inspecting the nature of extracted features. This seems relevant for both Fourier and Wavelet feature extractor and for <code>tsfel</code>. In the latter case, this can be done by adding an output information to the wrapper provided by <code>extract_tsfel_features.py</code> .

<b>Feedback and additional questions.</b>

All questions about the source code should be adressed to its author Alexandre Aksenov:
* GitHub: Alexandre-aksenov
* Email: alexander1aksenov@gmail.com
