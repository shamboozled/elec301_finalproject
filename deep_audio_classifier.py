import os
from matplotlib import pyplot as plt
import tensorflow as tf 
import tensorflow_io as tfio
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Activation, MaxPooling2D, Dropout
import pickle
import numpy as np
import pandas as pd


# inspired by: https://github.com/nicknochnack/DeepAudioClassification/blob/main/AudioClassification.ipynb

## CREATE DATASET
# all of these pickles are just numpy arrays containing the downsampled .wav data or labels
with open("downsampled_data.pickle", "rb") as f:
    X = pickle.load(f) # max size is 84351, sampling rate of 16000 Hz
with open("downsampled_test.pickle", "rb") as f:
    Q = pickle.load(f) # max size is 84351, sampling rate of 16000 Hz
with open("labels.pickle", "rb") as f:
    y = pickle.load(f)

## LABEL DATA
labeled_data = tf.data.Dataset.from_tensor_slices((X, y))

## SPECTROGRAM OF A SIGNAL
def specgram(signal, label):
    spectrogram = tf.signal.stft(signal, frame_length=256, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label

# signal, label = labeled_data.shuffle(buffer_size=1000).as_numpy_iterator().next()
# spectrogram, label = specgram(signal, label)
# plt.figure(figsize=(30,20))
# plt.imshow(tf.transpose(spectrogram)[0])
# plt.show()

## APPLY SPECGRAM TO ALL SIGNALS
specgram_data = labeled_data.map(specgram)
specgram_data = specgram_data.cache()
specgram_data = specgram_data.shuffle(buffer_size=1000)
specgram_data = specgram_data.batch(16)
specgram_data = specgram_data.prefetch(8)
print(len(specgram_data)) # length of data is 70

## SPLIT INTO TRAIN AND TEST SETS
train_size = int(len(specgram_data) * 0.5)
valid_size = len(specgram_data) - train_size
train = specgram_data.take(train_size)
valid = specgram_data.skip(train_size).take(valid_size)

## CREATE MODEL
model = Sequential()
model.add(Conv2D(8, (5, 5), strides=3, input_shape=(657, 129, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(8))
model.add(Activation('softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
    run_eagerly=True)

model.summary()

## TRAIN MODEL
hist = model.fit(train, epochs=20, verbose=1, validation_data=valid)

## SAVE MODEL
model.save("model_v7.h5")

## PLOT LOSS
plt.subplot(211)
plt.title('Loss')
plt.plot(hist.history['loss'], 'r')
plt.plot(hist.history['val_loss'], 'b')

plt.subplot(212)
plt.title('Accuracy')
plt.plot(hist.history['accuracy'], 'r')
plt.plot(hist.history['val_accuracy'], 'b')
plt.show()

#####################################################################

# # LOAD MODEL
# model = tf.keras.models.load_model("model_v7.h5")
# model.summary()

# ## MAKE PREDICTIONS
# shape = specgram(Q[0], 0)[0].shape
# temp = np.empty((315, shape[0], shape[1], shape[2]))

# for i in range(315):
#     temp[i] = specgram(Q[i], 0)[0]

# yhat = model.predict(temp)
# results = np.argmax(yhat, axis=1)

# ## FORMAT AND SAVE RESULTS AS CSV
# test_wav = os.listdir("test")

# labels_dict = {0:"neutral", 1:"calm", 2:"happy", 3:"sad", 4:"angry", 5:"fearful", 6:"disgust", 7:"surprise"}
# results_label = [labels_dict[results[idx]] for idx in range(len(results))]

# filenames = [test_wav[idx][:-4] for idx in range(len(test_wav))]

# submission_array = np.empty((len(filenames), 2), dtype=object)
# submission_array[:, 0] = filenames
# submission_array[:, 1] = results_label

# df = pd.DataFrame(submission_array)
# df.to_csv('spectrogram_CNN_v7.csv', index=False, header=['filename', 'label'])
