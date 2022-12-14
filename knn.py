from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import train_test_split
from metric_learn import LMNN
import numpy as np
import pandas as pd
import pickle
import soundfile as sf
import os
from scipy.fft import dct

# GET DATA IN NUMPY ARRAY
data_wav = os.listdir('data')
# max_file_size = 253053
# data = np.empty((len(data_wav) - 5, max_file_size))  # minus 5 because there are 5 dual channel samples
# labels = np.empty((len(data_wav) - 5))
#
# counter = 0
# for idx in range(len(data_wav) - 5):
#     file, fs = sf.read('data/' + data_wav[idx])
#     if len(file.shape) == 1:
#         data[idx] = np.pad(file, (0, max_file_size - file.size), 'constant')
#         if data_wav[idx].startswith('neutral'):
#             labels[idx] = 0
#         elif data_wav[idx].startswith('calm'):
#             labels[idx] = 1
#         elif data_wav[idx].startswith('happy'):
#             labels[idx] = 2
#         elif data_wav[idx].startswith('sad'):
#             labels[idx] = 3
#         elif data_wav[idx].startswith('angry'):
#             labels[idx] = 4
#         elif data_wav[idx].startswith('fearful'):
#             labels[idx] = 5
#         elif data_wav[idx].startswith('disgust'):
#             labels[idx] = 6
#         elif data_wav[idx].startswith('surprised'):
#             labels[idx] = 7
#
# print('finished getting data')
#
# X = np.nan_to_num(data)
# y = np.nan_to_num(labels).astype(int)
#
# with open("data.pickle", 'wb') as f:
#     pickle.dump(X, f)
# with open("labels.pickle", "wb") as g:
#     pickle.dump(y, g)

# GET TEST DATA IN NUMPY ARRAY
test_wav = os.listdir('test')
# max_file_size = 253053
# test = np.empty((len(test_wav), max_file_size))
#
# for idx in range(len(test_wav)):
#     file, fs = sf.read('test/' + test_wav[idx])
#     if len(file.shape) == 1:
#         test[idx] = np.pad(file, (0, max_file_size - file.size), 'constant')
#
# T = np.nan_to_num(test)
#
# with open("test.pickle", "wb") as f:
#     pickle.dump(T, f)

with open("data.pickle", 'rb') as f:
    X = pickle.load(f)
with open("labels.pickle", "rb") as g:
    y = pickle.load(g)
with open("test.pickle", "rb") as h:
    T = pickle.load(h)

# for idx in range(y.size):
#     if y[idx] not in [0, 1, 2, 3, 4, 5, 6, 7]:
#         print((idx, y[idx], labels[idx], data_wav[idx]))

print('set X and y')

X = dct(X)
T = dct(T)

# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

X_train, y_train, X_test = X, y, T

num_neighbors = 1120 // 8

# lmnn = LMNN(k=num_neighbors, learn_rate=1e-4)
# lmnn.fit(X_train, y_train)

model = knn(n_neighbors=num_neighbors, metric="cosine")
model.fit(X_train, y_train)

print('finished training model')

# print(model.score(X_test, y_test))  # accuracy of about 13-14%
# print(model.predict(X_test))  # turns out that calm generates the widest spread of signals
# print(y_train)

results = model.predict(X_test)
labels_dict = {0:"neutral", 1:"calm", 2:"happy", 3:"sad", 4:"angry", 5:"fearful", 6:"disgust", 7:"surprise"}
results_label = [labels_dict[results[idx]] for idx in range(len(results))]

filenames = [test_wav[idx][:-4] for idx in range(len(test_wav))]

submission_array = np.empty((len(filenames), 2), dtype=object)
submission_array[:, 0] = filenames
submission_array[:, 1] = results_label

df = pd.DataFrame(submission_array)
df.to_csv('kNN_DCT.csv', index=False, header=['filename', 'label'])
