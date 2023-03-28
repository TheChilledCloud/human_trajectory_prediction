
"""
Human Trajectory Prediction using Recurrent Nueral Network and Long Short-Term Memory

Model creation and training

@author: Ahmad Alfaisal
"""

import numpy as np
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
import math
   
#-------------------------------------------------------------------------------
# functions declerations
def creating_model(n_steps, n_features):

    model = Sequential()
    model.add(LSTM(64, activation='tanh', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(32, activation='tanh'))
    model.add(Dense(2))   
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    return model


#-------------------------------------------------------------------------------
# initializing variables
# constants
FRAMES_PER_SECOND = 9
N_FEATURES = 6 
BUFFER = 2

# model variables
num_seconds_past = 12
n_seconds_future = 3
model_number = '22'
n_steps = n_frames_past = FRAMES_PER_SECOND * num_seconds_past
n_frames_future = math.ceil(FRAMES_PER_SECOND * n_seconds_future)
min_track_len = n_frames_past + n_frames_future + BUFFER

# path varialbes
saving_path = r'C:\Users\ahmad\Desktop\thesis\Master Thesis\human motion prediction\prediction with drone dataset\testing model and data/'
loading_path = r'C:\Users\ahmad\Desktop\thesis\Master Thesis\human motion prediction\prediction with drone dataset\testing model and data/'

#-------------------------------------------------------------------------------
# creating model
model = creating_model(n_steps, N_FEATURES)
model.summary()

#loading model inputs
x_train = np.load(loading_path + 'x_train_test.npy')
y_train = np.load(loading_path + 'y_train_test.npy')
   
# training the model
history = model.fit(x_train, y_train, validation_split=0.1, shuffle=True, epochs=1, verbose=1)

# saving the model
model.save(saving_path + 'testing_model' + model_number)