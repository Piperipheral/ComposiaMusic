# Generative Neural Network for Music Composition Using Variational Autoencoder Architecture
### A Thesis By William Karunia (001201600001)

This is a python GUI along with a pre-trained Autoencoder model [trained using this code](https://colab.research.google.com/drive/1LWERpMD2wCbbixl2Ni27XPtp0wg5n0-9?usp=sharing) intended to be used by music composers in order to generate prompts to be used as ideas or starting points music compositions.

## Requirements
Users should have python 3.5 (3.7 was used during development) or later installed in their system.
The packages required for the application to run:
1. [Tensorflow](https://www.tensorflow.org/install)
2. [Keras](https://keras.io/)
3. [matplotlib](https://matplotlib.org/users/installing.html)
4. [numpy](https://numpy.org/install/)
5. [pygame](https://www.pygame.org/wiki/GettingStarted)
6. [pyaudio](https://pypi.org/project/PyAudio/)
7. [midiutil](https://pypi.org/project/MIDIUtil/)

It is recommended that user uses a [virtual environment](https://virtualenv.pypa.io/en/latest/) when installing the packages and attempting to run the python script. 

## Running Instructions:
1. Make sure you have python 3.5 or later installed in your system by typing `python --version` on your terminal or command prompt.
2. If using virtual environment, make sure to activate the virtual environment by typing `virtualenv **_your virtual environment path_**` and activating it by running the activation script `**_your virtual environment path_**/scripts/activate`. Note that a virtual environment with all the necessary packages pre-installed is included in this repository (/venv)
3. install all the necessary packages listed above.
4. on completion, type `python **_the application directory_**/init.py`.

## Usage:
The application should open and reveal as such:
![GUI Display](https://i.imgur.com/ikEzAv9.jpg)

1. To generate a new set of notes drag your cursor around the upper left corner box. The yellow dot should move to your cursor location. The yellow dot represents the position of the 2d vector within the latent space.
2. Once generated, press the space bar to play the composition.
