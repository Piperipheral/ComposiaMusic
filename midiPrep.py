import os
from mido import MidiFile, MidiTrack, Message
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras as k
from keras.models import *
from myGlobal import MAX_VECTOR_LENGTH, MAX_CUT, CUT_OFFSET
from keras.layers import *

tf.__version__


cats = ["forTrain", "forTest"]
theRolls = []
theTest = []

# note to self: try and reduce input vector size
def to_piano_roll(fName):
    midi = MidiFile(fName)
    notes = 127
    # to compress notes for input layer,
    # ignore notes which are most likely not going to be player (lowermost and uppermost notes)
    tempo = 500000
    seconds_per_beat = tempo / 1000000.0
    seconds_per_tick = seconds_per_beat / midi.ticks_per_beat
    velocities = np.zeros(notes)
    sequence = []
    theRollsIn = []
    for m in midi:
        ticks = int(np.round(m.time / seconds_per_tick))
        ls = [velocities.copy()] * ticks
        sequence.extend(ls)
        if m.type == 'note_on':
            velocities[m.note] = m.velocity
        elif m.type == 'note_off':
            velocities[m.note] = 0
        else:
            continue
    piano_roll = np.array(sequence).T
    numRolls = piano_roll.shape[1] // MAX_VECTOR_LENGTH
    if (numRolls > MAX_CUT):
        numRolls = MAX_CUT
    print (numRolls)
    for xMan in range(numRolls - CUT_OFFSET):
        x = xMan + CUT_OFFSET
        if piano_roll.shape[1] > MAX_VECTOR_LENGTH:
            piano_roll_padded = piano_roll[:127, x * MAX_VECTOR_LENGTH: x * MAX_VECTOR_LENGTH + MAX_VECTOR_LENGTH]
            print (piano_roll_padded.shape)
        else:
            piano_roll_padded = np.zeros((127, MAX_VECTOR_LENGTH))
            piano_roll_padded[:127, :piano_roll.shape[1]] = piano_roll
        theRollsIn.append(piano_roll_padded)
    return theRollsIn


for directory in cats:
    if directory == 'forTrain':
        print(os.walk(directory))
        for root, subdirs, files in os.walk("/Users/timothysutanto/PycharmProjects/Adaptivia/" + directory):
            for file in files:
                print(os.path.join(root, file))
                path = os.path.join(root, file)
                if not (path.endswith('.mid') or path.endswith('.midi')):
                    continue
                try:
                    print("attempting to convert midi file")
                    something = to_piano_roll(path)
                    theRolls.extend(something)
                finally:
                    print("success")
    else:
        print(os.walk(directory))
        for root, subdirs, files in os.walk("/Users/timothysutanto/PycharmProjects/Adaptivia/" + directory):
            for file in files:
                print(os.path.join(root, file))
                path = os.path.join(root, file)
                if not (path.endswith('.mid') or path.endswith('.midi')):
                    continue
                try:
                    print("attempting to convert midi file")
                    arrayMan = to_piano_roll(path)
                    theTest.extend(arrayMan)
                finally:
                    print("success")

theRolls = np.asarray(theRolls)

culling_upper = theRolls > 0
theRolls[culling_upper] = 1
theTest = np.asarray(theTest)
culling_upper = theTest > 0
theTest[culling_upper] = 1

np.save('theRolls.npy', theRolls)
np.save('theTest.npy', theTest)