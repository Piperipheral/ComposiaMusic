import os
from midiutil.MidiFile import MIDIFile
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras as k
from keras.models import *
from keras.layers import *

from myGlobal import MAX_VECTOR_LENGTH

theRolls = np.load('theRolls.npy')
theTest = np.load('theTest.npy')
tf.__version__


def toMidi(the_input, fname):
    MyMIDI = MIDIFile(1)
    track = 0
    time = 0
    channel = 0
    tickTime = .25
    MyMIDI.addTrackName(track, time, "Sample Track")
    MyMIDI.addTempo(track, time, 480)
    the_input = the_input.transpose()
    flag = np.zeros((127,))
    startNote = np.zeros((127,))
    myTick = 0
    for xMan in range(len(the_input)):
        myTick += tickTime
        for yMan in range(len(the_input[xMan])):
            if the_input[xMan][yMan] > 0:
                if flag[yMan - 1] == 0:
                    startNote[yMan - 1] = myTick
                flag[yMan - 1] += tickTime
            else:
                if flag[yMan - 1] != 0:
                    pitch = yMan
                    time = startNote[yMan - 1]
                    duration = flag[yMan - 1]
                    volume = 100
                    MyMIDI.addNote(track, channel, pitch, time, duration, volume)
                    print("appending: " + str(yMan) + " at " + str(time) + " for " + str(duration))
                    flag[yMan - 1] = 0
    binfile = open(fname, 'wb')
    MyMIDI.writeFile(binfile)
    binfile.close()

model_init = Sequential()
model_init.add(InputLayer(input_shape=(127, MAX_VECTOR_LENGTH,)))
model_init.add(TimeDistributed(Dense(64, activation='relu')))
model_init.add(Flatten())
model_init.add(Dense(32, activation='relu'))
model_init.add(Dense(64, activation='relu'))
model_init.add(Dense(127 * MAX_VECTOR_LENGTH, activation='sigmoid', name='decode'))
model_init.summary()
model_init.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

decoderModel = Sequential()
decoderModel.add(InputLayer(input_shape=(32,)))
decoderModel.add(Dense(64, activation="relu"))
decoderModel.add(Dense(127 * MAX_VECTOR_LENGTH, activation='sigmoid', name='decode'))
decoderModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
decoderModel.summary()

theThing = np.random.random_integers(0, 1000, (10, 32,))

superThing = np.asarray(theThing)
print (superThing.shape)
print (len(superThing))

for man in range(19):
    model_init.load_weights("weights/" + str(man + 1) + "00.h5")
    decoderModel.layers[0].set_weights(model_init.layers[3].get_weights())
    decoderModel.layers[1].set_weights(model_init.layers[4].get_weights())
    result = decoderModel.predict(superThing)
    normalizer = 10
    result = np.asarray(result).reshape(-1, 127, MAX_VECTOR_LENGTH)
    culling_lower = result < 0.8
    culling_upper = result >= 0.8
    result[culling_lower] = 0
    result[culling_upper] = 1

    index = 0
    for i in result:
        for j_index in range(len(i)):
            j = i[j_index]
            normalized = np.zeros((len(j),))
            for k in range(len(j)):
                if j[k] != 0:
                    for norm in range(normalizer):
                        if k + norm < len(j):
                            normalized[k + norm] = 1
            i[j_index] = normalized

    howMany = result.shape[0]
    resultComp = np.zeros((result.shape[1], result.shape[0] * MAX_VECTOR_LENGTH))
    for x in range(howMany):
        resultComp[:127, x * MAX_VECTOR_LENGTH: x * MAX_VECTOR_LENGTH + MAX_VECTOR_LENGTH] = result[x]
    toMidi(resultComp, "Results/testComp" + str(man) + ".midi")

fig, some = plt.subplots(nrows=2, ncols=3, figsize=(6, 10))
cur = 0
for x in some:
    intMan = 0
    for y in x:
        if cur == 0:
            y.imshow(result[intMan], aspect="auto")
        else:
            y.plot(theThing[intMan], 'r+')
        intMan += 1
    cur += 1

plt.show()
