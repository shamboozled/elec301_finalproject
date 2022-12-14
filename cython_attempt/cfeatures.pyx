import numpy as np
import emd
import antropy as ent
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
# got rid of double[:, ::1] type for X
def features(A, feature_matrix):
    cdef double[:, :] X = A
    cdef Py_ssize_t num_data = X.shape[0]
    cdef double[:, :] feature_matrix_view = feature_matrix

    cdef Py_ssize_t idx, counter
    # cdef double[:, :] imfs, normed_imfs, feature
    cdef double[:] residue, row, new_feature
    for idx in range(num_data):
        row = X[idx, :]
        imfs = emd.sift.sift(np.array(row), max_imfs=10).T
        residue = row - (np.sum(imfs, axis=0))
        normed_imfs = np.empty((8, 126527))
        normed_imfs[0:5] = imfs[0:5]
        normed_imfs[6] = np.add(imfs[6],imfs[7]) / 126527
        normed_imfs[7] = np.add(np.add(imfs[8], imfs[9]), residue) / 126527

        feature = np.empty((40))
        counter = 0
        for imf in normed_imfs:
            feature[counter] = ent.app_entropy(imf)
            feature[counter + 1] = ent.perm_entropy(imf)
            feature[counter + 2] = ent.sample_entropy(imf)
            feature[counter + 3] = ent.svd_entropy(imf)
            feature[counter + 4] = ent.spectral_entropy(imf, 24000)
            counter += 5

        new_feature = np.nan_to_num(feature)
        feature_matrix_view[idx, :] = new_feature
