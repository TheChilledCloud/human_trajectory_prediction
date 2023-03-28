
"""
Human Trajectory Prediction using Recurrent Nueral Network and Long Short-Term Memory

Data cleaning and preprocessing

@author: Ahmad Alfaisal

"""

# loading the required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
import pickle

#-------------------------------------------------------------------------------
# functions declerations
def drop_column(df, column):
    
	df = df.drop(column, axis = 1)

	return df
 
def filter_transport_type(df, trackId_list, column):
    
    df_ped_loc = pd.DataFrame()
    for i in trackId_list:
        templist = df.loc[df[column] == i]
        df_ped_loc = df_ped_loc.append(templist, ignore_index=True)
        
    return df_ped_loc

def split_train_test(df_ped_loc_scaled, trackId):
    
    
    num_training_tracks = int(len(trackId) * 0.98)
    trackId_value = trackId[num_training_tracks]
    index = df_ped_loc_scaled.loc[df_ped_loc_scaled['trackId'] == trackId_value]
    index = index.index[0]
    training_df = df_ped_loc_scaled.iloc[:index]
    testing_df = df_ped_loc_scaled.iloc[index:]

    return training_df, testing_df

def split_x_y(df, n_sec_future, n_sec_past):

    x = []
    y = [] 
    for i in range(n_sec_past, len(df) - n_sec_future+1):
        x.append(df[i - n_sec_past:i, 0:])
        y.append(df[i + n_sec_future -1 : i + n_sec_future, 0:])

    return x, y
    
def min_max_scaler(df):

    dataframe = pd.DataFrame(df)
    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(dataframe)

    return scaled_df, scaler

def reshaping_data(x_train, y_train):
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = x_train.reshape(x_train.shape[0],1,x_train.shape[1])
    #print(x.shape)
    return x_train, y_train

def selecting_features(df):
    
    df = pd.DataFrame(df)
    df = drop_column(df, column='recordingId')
    df = drop_column(df, column='trackLifetime')
    df = drop_column(df, column='heading')
    df = drop_column(df, column='width')
    df = drop_column(df, column='length')
    df = drop_column(df, column='lonVelocity')
    df = drop_column(df, column='latVelocity')
    df = drop_column(df, column='lonAcceleration')
    df = drop_column(df, column='latAcceleration')

    return df

def selecting_pedestrians(df_ped, df_tracks):
    
    df_ped = pd.DataFrame(df_ped)
    df_ped = df_ped.loc[df_ped['class'] == 'pedestrian']
    trackId = list(df_ped['trackId'])
    
    # keeping only pedestrians data and removing the rest
    df_pedestrians_tracks = filter_transport_type(df_tracks, trackId, column='trackId')

    return df_pedestrians_tracks, trackId

def prepare_for_scaling(df):
    
    df = pd.DataFrame(df)
    trackId_column = df['trackId']
    frame_column = df['frame']
    
    # removing the trackId and frame columns from df_ped_loc dataframe
    df = drop_column(df, column='trackId')
    df = drop_column(df, column='frame')

    return df, trackId_column, frame_column

def save_list_to_file(myList, saving_path):
    
    with open(saving_path, 'wb') as file:
        pickle.dump(myList, file)
        
#-------------------------------------------------------------------------------
# initializing variables
# constants
FRAMES_PER_SECOND = 9
BUFFER = 2

# variables
# tracks counting variables
number_trainable_tracks = 0
number_tracks_overall = 0
number_testable_tracks = 0

# data splitting variables
n_seconds_past = 1
n_seconds_future = 0.5
model_number = '12'
n_frames_past = FRAMES_PER_SECOND * n_seconds_past
n_frames_future = math.ceil(FRAMES_PER_SECOND * n_seconds_future)
min_track_len = n_frames_past + n_frames_future + BUFFER

# path varialbes
saving_path = r'C:\Users\ahmad\Desktop\thesis\Master Thesis\human motion prediction\prediction with drone dataset\model'+ model_number +'_testing_results/'
loading_path = r'C:\Users\ahmad\Desktop\thesis\Master Thesis\human motion prediction\prediction with drone dataset/'

# lists declerations
x_train_list = []
y_train_list = []
tracksPathList = []
metaPathList = []
testing_tracks_list = []
nameList = ['00','01','02','03','04','05','06','07','08','09','10',
            '11','12','13','14','15','16','17','18','19','20','21',
            '22','23','24','25','26','27','28','29','30','31','32']

#-------------------------------------------------------------------------------
# loading files paths names into lists
for i in nameList:
  tracksPath = loading_path + i + '_tracks.csv'
  metaPath = loading_path + i + '_tracksMeta.csv'
  tracksPathList.append(tracksPath)
  metaPathList.append(metaPath)

#len(nameList)
for i in range(len(nameList)):
        
    #loading the ith file into a dataframe
    df_tracks = pd.read_csv(tracksPathList[i])
    
    # removing unrequired features from the dataframe
    df_tracks = selecting_features(df_tracks)
    
    # loading the pedestrians trackId data
    df_pedestrian = pd.read_csv(metaPathList[i])
    
    #filterign out the pedestrians tracks from the tracks dataframe and coping them to new dataframe
    df_pedestrians_tracks, trackId = selecting_pedestrians(df_pedestrian, df_tracks)
        
    # preparing the dataframe to be scaled, i.e. remove the frame and track id columns from the dataframe so we can scale the data inside it
    df_pedestrians_tracks, trackId_column, frame_column = prepare_for_scaling(df_pedestrians_tracks)
    
    # scaling the data between 0,1
    df_pedestrians_tracks, scaler = min_max_scaler(df_pedestrians_tracks)
    
    #converting the scaled data from np.array to dataframe
    df_pedestrians_tracks = pd.DataFrame(df_pedestrians_tracks)
    
    # adding the tracks ids and frame numbers back to the dataframe
    df_pedestrians_tracks['trackId'] = trackId_column
    df_pedestrians_tracks['frame'] = frame_column
    
    # naming the columns of the df_ped_loc_scaled dataframe
    df_pedestrians_tracks.columns=['xCenter','yCenter', 'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration', 'trackId', 'frame']
    
    # splitting the data into training and testing sets
    training_df, testing_df = split_train_test(df_pedestrians_tracks, trackId)
    
    # finding how many tracks we have in each file and overall
    number_tracks_overall = number_tracks_overall + len(trackId)
    print('number of tracks in this file is: ',len(trackId))
    
    #counting number of tracks and their lengths in testing set
    for track in trackId:
        ith_track_testing_data = testing_df.loc[testing_df['trackId']==track]
        if (len(ith_track_testing_data) == 0) or (len(ith_track_testing_data)/3 < min_track_len):
            continue
        testing_tracks_list.append(tracksPathList[i])
        testing_tracks_list.append(metaPathList[i])
        testing_tracks_list.append(track)
        testing_tracks_list.append(len(ith_track_testing_data))
        number_testable_tracks = number_testable_tracks + 1

    # taking Ith track data and converting it to np array for both testing and traininig data
    for track in trackId:
        
        # selecting the required track from the training dataframe and coping its data to a new dataframe
        ith_track_training_data = training_df.loc[training_df['trackId']==track]
        
        # devide the number of frames by 3, so we have 8.3 frames per second only (round up to 9)
        ith_track_training_data = ith_track_training_data.iloc[0:len(ith_track_training_data):3]
            
        #check if the track has sufficient frames for training, if not, select next track 
        if len(ith_track_training_data) < min_track_len:
            print('the track which is less than min_track_len in training is: ',track)
            continue

        #count number of tracks the model is training on
        number_trainable_tracks = number_trainable_tracks + 1

        #convert to the track data to array
        ith_track_training_data = np.array(ith_track_training_data)

        # splitting the track data into input (x_train) and output (y_train) data
        x_train_track, y_train_track = split_x_y(ith_track_training_data, n_frames_future, n_frames_past)
        
        # converting x_train and y_train to arrays and reshaping them
        x_train_track, y_train_track = np.array(x_train_track), np.array(y_train_track)
        y_train_track = y_train_track.reshape(y_train_track.shape[0],y_train_track.shape[2])
        
        # removing the trackId and frame columns from x_train, and keeping only x and y coordinates in y_train
        x_train_track = np.delete(x_train_track, 7, 2)
        x_train_track = np.delete(x_train_track, 6, 2)
        y_train_track = np.delete(y_train_track, 7, 1)
        y_train_track = np.delete(y_train_track, 6, 1)
        y_train_track = np.delete(y_train_track, 5, 1)
        y_train_track = np.delete(y_train_track, 4, 1)
        y_train_track = np.delete(y_train_track, 3, 1)
        y_train_track = np.delete(y_train_track, 2, 1)
        
        # adding the input and output data of all the files to 2 lists
        x_train_list.append(x_train_track)
        y_train_list.append(y_train_track)
        

# concatenating the training data in an array
x_train = np.array(x_train_list[0])
for i in range(len(x_train_list)):
    x_train = np.concatenate((x_train, x_train_list[i]), axis = 0)

y_train = np.array(y_train_list[0])
for i in range(len(y_train_list)):
    y_train = np.concatenate((y_train, y_train_list[i]), axis = 0)
    
# saving the training arrays
np.save(saving_path + 'x_train_test.npy', x_train)
np.save(saving_path + 'y_train_test.npy', y_train)

# saving the testing tracks list
save_list_to_file(testing_tracks_list, saving_path + 'testing_tracks_list.pkl')
