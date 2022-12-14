import time
import numpy as np
import soundfile as sf
import pickle
from scipy import fft
import matplotlib.pyplot as plt

with open("test.pickle", 'rb') as f:
    X = pickle.load(f)

downsampled = X[:, ::3]
print(X.shape)
print(downsampled.shape)

with open("downsampled_test.pickle", 'wb') as g:
    pickle.dump(downsampled, g)

# signal, fs = sf.read('downsampled_angry000.wav')
# signal_dct = fft.dct(signal)

# num_frequencies_to_keep = 20000
# important_dct_indices = np.argsort(np.abs(signal_dct))[-num_frequencies_to_keep:]
# reduced_dct = np.zeros(signal_dct.shape)
# reduced_dct[important_dct_indices] = signal_dct[important_dct_indices]

# recovered_signal = fft.idct(reduced_dct)

# sf.write('compressed_angry000.wav', recovered_signal, fs)

# plt.subplot(231)
# plt.plot(range(signal.size), signal)
# plt.subplot(232)
# plt.plot(range(recovered_signal.size), recovered_signal)
# plt.subplot(233)
# plt.plot(range(X[idx].size), X[idx])
# plt.subplot(234)
# plt.plot(range(signal_dct.size), signal_dct)
# plt.subplot(235)
# plt.plot(range(reduced_dct.size), reduced_dct)
# plt.subplot(236)
# plt.plot(range(X[idx].size), fft.dct(X[idx]))
# plt.show()
