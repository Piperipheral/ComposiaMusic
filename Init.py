import os
from mido import MidiFile, MidiTrack, Message
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras as k
from keras.models import *
from keras.optimizers import *
from keras.layers import *
from keras.regularizers import *
from myGlobal import MAX_VECTOR_LENGTH, NUM_EPOCHS, LEARNING_RATE

tf.__version__
theRolls = np.load('theRolls.npy')
theTest = np.load('theTest.npy')
X_train = theRolls.reshape(len(theRolls), np.prod(theRolls.shape[1:]))

mc = k.callbacks.ModelCheckpoint('weights/{epoch:d}.h5',
                                     save_weights_only=True, period=100)

print (X_train.shape)

print (theTest.shape)

np.random.seed(1)
# attempting to model a basic neural network
noise = np.random.normal(0, 10, (X_train.shape[0], X_train.shape[1]))
X_train_noisy = X_train + noise

model = Sequential()
model.add(InputLayer(input_shape=(127, MAX_VECTOR_LENGTH,)))
model.add(TimeDistributed(Dense(64, activation='relu')))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(127 * MAX_VECTOR_LENGTH, activation='sigmoid', name='decode'))
model.summary()

print (model.layers[2].output)

adamSlow = adam(lr= LEARNING_RATE)
model.compile(optimizer=adamSlow, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(theRolls, X_train, epochs=NUM_EPOCHS, callbacks=[mc])

X_Test = theTest.reshape(len(theTest), np.prod(theTest.shape[1:]))
result = model.predict(theRolls)


print (result.shape)
result = np.asarray(result).reshape(-1, 127, MAX_VECTOR_LENGTH)

fig, some = plt.subplots(nrows=2, ncols=3, figsize=(6, 10))
cur = 0
for x in some:
    intMan = 0
    for y in x:
        if cur == 0:
            y.imshow(result[intMan], aspect="auto")
        else:
            y.imshow(theTest[intMan], aspect="auto")
        intMan += 1
    cur += 1
plt.show()

