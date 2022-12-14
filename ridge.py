from sklearn.linear_model import RidgeClassifier
import pickle
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np

with open("data.pickle", 'rb') as f:
    X = pickle.load(f)
with open("labels.pickle", "rb") as g:
    y = pickle.load(g)
with open("test.pickle", "rb") as h:
    T = pickle.load(h)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

print('finished loading data')

ridge = RidgeClassifier(tol=0.00001)

ridge.fit(X, y)
results = ridge.predict(T)

labels_dict = {0:"neutral", 1:"calm", 2:"happy", 3:"sad", 4:"angry", 5:"fearful", 6:"disgust", 7:"surprise"}
results_label = [labels_dict[results[idx]] for idx in range(len(results))]

test_wav = os.listdir('test')
filenames = [test_wav[idx][:-4] for idx in range(len(test_wav))]

submission_array = np.empty((len(filenames), 2), dtype=object)
submission_array[:, 0] = filenames
submission_array[:, 1] = results_label

df = pd.DataFrame(submission_array)
df.to_csv('ridge.csv', index=False, header=['filename', 'label'])
