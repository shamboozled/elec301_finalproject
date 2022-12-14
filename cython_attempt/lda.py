from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import time
from multiprocessing import Process
import emd
import antropy as ent
from functools import partial
from cfeatures import features


# def features(X, feature_matrix):
#     get_features_partial = partial(get_features, X, feature_matrix)
#     for idx in range(X.shape[0]):
#         get_features_partial(idx)

# def get_features(X, feature_matrix, idx):
#     imfs = emd.sift.sift(X[idx], max_imfs=10).T
#     residue = X[idx] - (np.sum(imfs, axis=0))
#     normed_imfs = np.empty((8, 126527))
#     normed_imfs[0:5] = imfs[0:5]
#     normed_imfs[6] = (imfs[6] + imfs[7]) / 126527
#     normed_imfs[7] = (imfs[8] + imfs[9] + residue) / 126527

#     feature = np.empty((1, 40))
#     counter = 0
#     for imf in normed_imfs:
#         feature[0, counter] = ent.app_entropy(imf)
#         feature[0, counter + 1] = ent.perm_entropy(imf)
#         feature[0, counter + 2] = ent.sample_entropy(imf)
#         feature[0, counter + 3] = ent.svd_entropy(imf)
#         feature[0, counter + 4] = ent.spectral_entropy(imf, 24000)
#         counter += 5

#     feature_matrix[idx] = np.nan_to_num(feature)

with open("downsampled_data.pickle", 'rb') as f:
    X = pickle.load(f)
with open("labels.pickle", "rb") as g:
    y = pickle.load(g)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

print('finished loading data')

test = X_train[0:1]
test_return = np.zeros((test.shape[0], 40))

start = time.perf_counter()

features(test, test_return)

print(time.perf_counter() - start)

print(test_return.shape)
print(test_return)

# if __name__ == '__main__':
#     p1 = Process(target=features, args=(test[::3], test_return[::3]))
#     p2 = Process(target=features, args=(test[1::3], test_return[1::3]))
#     p3 = Process(target=features, args=(test[2::3], test_return[2::3]))

#     p1.start()
#     p2.start()
#     p3.start()

#     p1.join()
#     p2.join()
#     p3.join()

#     print(time.perf_counter() - start)

#     print(test_return)

# X_train_processed = features(np.array([X_train[0:1]]))
# X_test_processed = features(X_test)

# with open("X_train_imf_data.pickle", 'wb') as f:
#     pickle.dump(X_train_processed, f)
# with open("X_test_imf_data.pickle", "wb") as g:
#     pickle.dump(X_test_processed, g)
