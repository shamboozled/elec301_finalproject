import numpy as np
import emd
import antropy as ent
import multiprocessing as mp
from functools import partial


def features(X, feature_matrix):
    get_features_partial = partial(get_features, X, feature_matrix)
    for idx in range(X.shape[0]):
        get_features_partial(idx)


def get_features(X, feature_matrix, idx):
    imfs = emd.sift.sift(X[idx], max_imfs=10).T
    residue = X[idx] - (np.sum(imfs, axis=0))
    normed_imfs = np.empty((8, 126527))
    normed_imfs[0:5] = imfs[0:5]
    normed_imfs[6] = (imfs[6] + imfs[7]) / 126527
    normed_imfs[7] = (imfs[8] + imfs[9] + residue) / 126527

    feature = np.empty((1, 40))
    counter = 0
    for imf in normed_imfs:
        feature[0, counter] = ent.app_entropy(imf)
        feature[0, counter + 1] = ent.perm_entropy(imf)
        feature[0, counter + 2] = ent.sample_entropy(imf)
        feature[0, counter + 3] = ent.svd_entropy(imf)
        feature[0, counter + 4] = ent.spectral_entropy(imf, 24000)
        counter += 5

    feature_matrix[idx] = np.nan_to_num(feature)
