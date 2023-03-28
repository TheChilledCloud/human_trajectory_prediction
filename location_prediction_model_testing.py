"""
Human Trajectory Prediction using Recurrent Nueral Network and Long Short-Term Memory

Model testing

@author: Ahmad Alfaisal

"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle
import math
import os

def drop_column(df, column):
    
	df = df.drop(column, axis = 1)

	return df

 
def filter_transport_type(df,trackId_list,column):
    
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

def inverse_transform(arr, scaler):
    
    inverted_data = scaler.inverse_transform(arr)

    return inverted_data

def final_displacement_error(pred,true):
    
    r = np.sqrt((true[:,0:1]-pred[:,0:1])**2 + (true[:,1:2]-pred[:,1:2])**2)
    sum = 0
    for element in r:
        sum += element
    fde = sum/len(r)
    
    return fde, r

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

def selecting_pedestrians(df, df_new):
    
    df_meta = pd.DataFrame(df)
    df_ped = df_meta.loc[df_meta['class'] == 'pedestrian']
    trackId = list(df_ped['trackId'])
    
    # keeping only pedestrians data and removing the rest
    df_ped_loc = filter_transport_type(df_new,trackId,column='trackId')


    return df_ped_loc, trackId

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

def load_list_from_file(loading_path):
    
    with open(loading_path, 'rb') as file:
        
        return pickle.load(file)
    
def create_bar_graph(x, y, titel):
    
    plt.bar(x, y,  width=0.3, label='observed')
    plt.title(titel)
    plt.ylabel('points percentage %')
    plt.xlabel('error (cm)')
    plt.savefig(saving_path + 'model' + str(model_number) + titel + '.png')
    plt.show() 

def save_list_to_txt(myList, saving_path):
    
    with open(saving_path, 'w') as output:
        output.write(str(myList))
        
def average_of_list(myList):
    
    average_fde = 0
    for i in range(1, len(myList),2):
        average_fde = average_fde + float(myList[i])
    average_fde = average_fde/(len(myList)/2)
    
    return average_fde

def create_line_graph(x1, y1, label1, x2, y2, label2, x3, y3, label3, saving_path):
    
    plt.plot(x1, y1, 'b', label= label1)
    plt.plot(x2, y2, 'g', label= label2)
    plt.plot(x3, y3, 'r', label= label3)
    plt.scatter(x1[0], y1[0], marker='x', color='black', label='starting point')
    plt.legend()
    plt.ylabel('y (m)')
    plt.xlabel('x (m)')
    plt.legend() 
    plt.savefig(saving_path)
    plt.show()

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

# tracks counting variables
number_testable_tracks = 0
number_untestable_tracks = 0

# displacement error varialbs
less_than_10cm = 0
less_than_20cm = 0
less_than_30cm = 0
less_than_40cm = 0
less_than_50cm = 0
less_than_60cm = 0
less_than_70cm = 0
less_than_80cm = 0
less_than_90cm = 0
less_than_100cm = 0
greater_than_100cm = 0

# path varialbes
saving_path = r'C:\Users\ahmad\Desktop\thesis\Master Thesis\human motion prediction\prediction with drone dataset\model' + model_number +'_results/'
loading_path = r'C:\Users\ahmad\Desktop\thesis\Master Thesis\human motion prediction\prediction with drone dataset/'

# lists declerations
displacement_type_count = []
displacement_len = ['<10','<20','<30','<40','<50','<60','<70','<80','<90','<100','>100']
error_list = []
tracksPathList = []
metaPathList = []
nameList = ['00','01','02','03','04','05','06','07','08','09','10',
            '11','12','13','14','15','16','17','18','19','20','21',
            '22','23','24','25','26','27','28','29','30','31','32']

#-------------------------------------------------------------------------------
# loading the model, the testing track list and the files
# loading the model
model = load_model(loading_path + 'location_prediction_model' + model_number)

# loading the testing tracks list
testing_tracks_list =  load_list_from_file(loading_path + 'testing_tracks_list.pkl')

# create a new directory to save files to
os.makedirs(saving_path)

# loading files paths names into lists
for i in nameList:
  tracksPath = loading_path + i + '_tracks.csv'
  metaPath = loading_path + i + '_tracksMeta.csv'
  tracksPathList.append(tracksPath)
  metaPathList.append(metaPath)

#-------------------------------------------------------------------------------
# looping over the testing tracks to test the model performance track by track and produce comparison graphs
for i in range(0, len(testing_tracks_list), 4):
    
    #loading the ith file into a dataframe
    track_path = i            
    df_tracks = pd.read_csv(testing_tracks_list[track_path])
    
    # removing unrequired features from the dataframe
    df_tracks = selecting_features(df_tracks)
    
    # loading the pedestrians trackId data
    meta_path = i+1
    df_pedestrian = pd.read_csv(testing_tracks_list[meta_path])
    
    # filterign out the pedestrians tracks from the tracks dataframe and coping them to new dataframe
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
    
    # selecting the required track from the testing dataframe and copying its data to a new dataframe
    item = i + 2
    ith_track_testing_data = testing_df.loc[testing_df['trackId']==testing_tracks_list[item]]
    
    # devide the number of frames by 3, so we have 8.3 frames per second only (round up to 9)
    ith_track_testing_data = ith_track_testing_data.iloc[0:len(ith_track_testing_data):3]
    
    # converting ith_track_testing to np array 
    ith_track_testing_data = np.array(ith_track_testing_data)
    
    #check if the track has sufficient frames for testing, if not, select next track 
    if len(ith_track_testing_data) < min_track_len:
        #print('the track which is less than 130 frames in testing is: ',testing_tracks_list[item])
        number_untestable_tracks = number_untestable_tracks + 1
        continue
    
    # splitting the track testing data into input (x_test) and output (y_test) data
    x_test, y_test = split_x_y(ith_track_testing_data, n_frames_future, n_frames_past)
    
    # converting x_test and y_test to np array and reshaping them
    x_test, y_test = np.array(x_test), np.array(y_test)
    y_test = y_test.reshape(y_test.shape[0],y_test.shape[2])
    
    # removing the trackId and frame columns from x_test, and keeping only x and y coordinates in y_test
    x_test = np.delete(x_test, 7, 2)
    x_test = np.delete(x_test, 6, 2)
    y_test = np.delete(y_test, 7, 1)
    y_test = np.delete(y_test, 6, 1)
    y_test = np.delete(y_test, 5, 1)
    y_test = np.delete(y_test, 4, 1)
    y_test = np.delete(y_test, 3, 1)
    y_test = np.delete(y_test, 2, 1)
    
    predictions = model.predict(x_test, verbose=1)
    
    # changing the shape of predictions and y_test, in order to inverse tranform them
    # creating a new column to be added to the predictions
    new_column = predictions[:,0:2]
    
    # adding 4 more columns to the predictions, so total shape is (,6)
    predictions_copies = np.append(predictions, new_column, axis=1)
    predictions_copies = np.append(predictions_copies, new_column, axis=1)
    
    # inverse transforming the prediction data
    predictions = inverse_transform(predictions_copies, scaler)[:,0:2]
   
    # adding 4 more columns to the y_test, so total shape is (,6)
    y_test_copies = np.append(y_test, new_column, axis=1)
    y_test_copies = np.append(y_test_copies, new_column, axis=1)
    
    # inverse transforming the ground truth data
    y_test = inverse_transform(y_test_copies, scaler)[:,0:2]
    
    # creating a new array for plotting the observed data and changing its shape to inverse transform it
    # creating a new column to be added to the predictions
    new_column = ith_track_testing_data[:(min_track_len-BUFFER),0:2]
    
    # adding 4 more columns to the observed data, so total shape is (,6)
    x_test_copies = np.append(new_column, new_column, axis=1)
    x_test_copies = np.append(x_test_copies, new_column, axis=1)
    
    # inverse transforming the ovserved data
    observed = inverse_transform(x_test_copies, scaler)[:,0:2]
    
    #plotting the observed, ground truth and predicted track
    create_line_graph(observed[:,0], observed[:,1], 'observed', predictions[:,0], predictions[:,1], 'predicted', y_test[:,0], y_test[:,1], 'true',
                      saving_path + 'model' + str(model_number) + '_path' + str(i) + '_track' + str(testing_tracks_list[item]) + '.png')
        
    #calculate average displacement error
    fde, trajectories_dispalcements = final_displacement_error(predictions,y_test)
    
    #add errors to errors list
    error_list.append('track' + str(testing_tracks_list[item]))
    error_list.append(fde)   

    for element in trajectories_dispalcements:
        if element <= 0.1:
            less_than_10cm +=1
        elif element <= 0.2:
            less_than_20cm +=1
        elif element <= 0.3:
            less_than_30cm +=1
        elif element <= 0.4:
            less_than_40cm +=1
        elif element <= 0.5:
            less_than_50cm +=1
        elif element <= 0.6:
            less_than_60cm +=1
        elif element <= 0.7:
            less_than_70cm +=1
        elif element <= 0.8:
            less_than_80cm +=1
        elif element <= 0.9:
            less_than_90cm +=1
        elif element <= 1.00:
            less_than_100cm +=1
        else:
            greater_than_100cm +=1   


# creating a list to house all error distributions
displacement_type_count = [less_than_10cm,less_than_20cm,less_than_30cm,less_than_40cm,less_than_50cm,less_than_60cm,less_than_70cm,less_than_80cm,less_than_90cm,less_than_100cm,greater_than_100cm]

# calculate total number of error points
total_error_points = sum(displacement_type_count)

# compute the percentage of points in each category
percentages_error_points = [100.0 * displacement_type_count / total_error_points for displacement_type_count in displacement_type_count]

# Creating a bar plot
create_bar_graph(displacement_len, percentages_error_points, 'FDE points distribution Graph')

# saving the displacement error count list to a text file
save_list_to_txt(percentages_error_points ,saving_path + 'model' + model_number +'_FDE_distribution.txt')

# saving the displacement error count list to a pickle file
save_list_to_file(percentages_error_points ,saving_path + 'model' + model_number +'__FDE_distribution.txt')

# finding the average displacement error of all testing tracks
average_fde = average_of_list(error_list)  

#append average_fde to error list
error_list.append(average_fde) 

# saving the error list to a text file
save_list_to_txt(error_list ,saving_path + 'model' + model_number +'_error_list.txt')

# saving the list to a pickle file
save_list_to_file(error_list ,saving_path + 'model' + model_number +'_error_list.pkl')

