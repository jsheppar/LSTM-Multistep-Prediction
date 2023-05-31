import os
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from matplotlib import pyplot as plt
from tensorflow.keras.models import model_from_json

import tensorflow.compat.v1 as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm.auto import trange
from tqdm.keras import TqdmCallback

batch = None
n_units = 32

def rescale_no_pp(feature_data, test_targets, predictions, scaling_object_features, scaling_object_targets, index):
    '''Flattens and rescales test and prediction data back to the original scale.
    Given that the test data and predictions do not have the same shape as the original feature data, we need 
    to "pad" these two datasets with the original column numbers of the feature data, 
    as well as have the test and prediction data occupy the same positions of their respective 
    target data columns so the rescale is done properly. 
    The below code includes one way to correctly do this padding.
    
    INPUTS: training or test feature data (it doesn't matter--we just need the same number of columns)
    test targets, and predictions, all in 3D tensor form. Also, the scaling object used 
    for the original transformation'''
    
    #flatten predictions and test data
    predict_flat = predictions.reshape(predictions.shape[0]*predictions.shape[1], predictions.shape[2])
    y_test_flat = test_targets.reshape(test_targets.shape[0]*test_targets.shape[1], test_targets.shape[2])
        
    #We now have the correct dimensions, so we can FINALLY rescale
    y_test_rescale = scaling_object_targets.inverse_transform(y_test_flat)
    preds_rescale = scaling_object_targets.inverse_transform(predict_flat)
    
    preds_rescale = pd.DataFrame(preds_rescale, index=index)
    y_test_rescale = pd.DataFrame(y_test_rescale, index=index)
       
    #before we return the dataframes, check and see if predictions or test data have null values.
    if preds_rescale.isnull().values.any()==True:
        print('Keras predictions have NaN values present. Deleting...')
        print('Current shape: ' + str(preds_rescale.shape))
        nans = np.argwhere(np.isnan(preds_rescale.values)) #find nulls
        #delete make sure values are deleted from both
        y_test_rescale = np.delete(y_test_rescale.values, nans,axis=0)
        preds_rescale = np.delete(preds_rescale.values, nans, axis=0)
        index = np.delete(index, nans, axis=0)
        #turn back into dataframe in case next condition is also true
        preds_rescale = pd.DataFrame(preds_rescale, index=index) 
        y_test_rescale = pd.DataFrame(y_test_rescale, index=index)
        print('New Shape: ' + str(preds_rescale.shape))
        
    
    # if y_test_rescale.isnull().values.any()==True:
    #     print('Test data still have NaN values present. Deleting...')
    #     print('Current shape: ' + str(y_test_rescale.shape))
    #     nans = np.argwhere(np.isnan(y_test_rescale.values)) 
    #     #same as above
    #     y_test_rescale = np.delete(y_test_rescale.values, nans,axis=0)
    #     preds_rescale = np.delete(preds_rescale.values, nans, axis=0)
    #     index = np.delete(index, nans, axis=0)
    #     # make into DataFrame this time to guarantee the below return statement won't spit an error
    #     y_test_rescale = pd.DataFrame(y_test_rescale,index=index)
    #     preds_rescale = pd.DataFrame(preds_rescale,index=index)
    #     print('New shape: ' + str(y_test_rescale.shape))
    
    #print('test data new shape: ' + str(y_test_rescale.shape))
    #print('prediction new shape: ' + str(preds_rescale.shape))
    return y_test_rescale, preds_rescale

def rescale_w_pp(feature_data, test_targets, predictions, scaling_object, index):
    '''Flattens and rescales test and prediction data back to the original scale.
    Given that the test data and predictions do not have the same shape as the original feature data, we need 
    to "pad" these two datasets with the original column numbers of the feature data, 
    as well as have the test and prediction data occupy the same positions of their respective 
    target data columns so the rescale is done properly. 
    The below code includes one way to correctly do this padding.
    
    INPUTS: training or test feature data (it doesn't matter--we just need the same number of columns)
    test targets, and predictions, all in 3D tensor form. Also, the scaling object used 
    for the original transformation'''
    print(np.shape(feature_data))
    print(np.shape(test_targets))
    print(np.shape(predictions))
    
    #flatten predictions and test data
    predict_flat = predictions.reshape(predictions.shape[0]*predictions.shape[1], predictions.shape[2])
    y_test_flat = test_targets.reshape(test_targets.shape[0]*test_targets.shape[1], test_targets.shape[2])
    
    print(np.shape(predict_flat))
    
    #flatten the features dataframe. This has the dimensions we want.
    flattened_features = pd.DataFrame(feature_data.reshape(feature_data.shape[0]*feature_data.shape[1],
                                                          feature_data.shape[2]))
    print(np.shape(flattened_features))
    #if we want to predict long sequences these will likely be longer than the length of the features.
    #we just add some zeros here to pad out the feature length to match. We'll then convert these to NaN's
    #so they're ignored. Again, we just want to inherient the
    #structure of the feature data--not the values. This is the most foolproof method
    #I've found through trial and error.
    
    if len(flattened_features) < len(y_test_flat):
        print('Length of targets exceeds length of features. Now padding...\n')
        extra_rows = pd.DataFrame(np.zeros((len(y_test_flat), flattened_features.shape[1])))
        flattened_features = pd.concat([flattened_features, extra_rows], axis=0)
        flattened_features[flattened_features==0]=np.nan
        
    #make a start column, this is the index where we begin to repopulate the target cols with the 
    #data we want to rescale
    start_col = feature_data.shape[2]-test_targets.shape[2]
    total_col = feature_data.shape[2]
    
    #make trimmed feature copies of equal length as the test data and predictions lengths, 
    #and leave out the original target data... we will replace these cols with the test and prediction data
    flattened_features_test_copy = flattened_features.iloc[:len(y_test_flat), :start_col]
    flattened_features_pred_copy = flattened_features.iloc[:len(predict_flat), :start_col]
    #print((flattened_features_pred_copy.values))
    
    for i in range(start_col, total_col):
        #reassign targets cols
        flattened_features_test_copy[i] = y_test_flat[:, i-start_col] 
        flattened_features_pred_copy[i] = predict_flat[:, i-start_col]
        #by specifying 'i - start col', we are making sure the target column being 
        #repopulated is the matching target taken from the test data or predictions.
        #Ex: if the start col is 4, then we want to assign the first column of the test and pred data--
        #this is index 0, and 4-4 = 0.
        
    #We now have the correct dimensions, so we can FINALLY rescale
    y_test_rescale = scaling_object.inverse_transform(flattened_features_test_copy)
    preds_rescale = scaling_object.inverse_transform(flattened_features_pred_copy)
    
    #just grab the target cols.
    y_test_rescale = y_test_rescale[:, start_col:]
    preds_rescale = preds_rescale[:, start_col:]
    
    preds_rescale = pd.DataFrame(preds_rescale, index=index)
    y_test_rescale = pd.DataFrame(y_test_rescale, index=index)
    
    
    #before we return the dataframes, check and see if predictions or test data have null values.
    if preds_rescale.isnull().values.any()==True:
        print('Keras predictions have NaN values present. Deleting...')
        print('Current shape: ' + str(preds_rescale.shape))
        nans = np.argwhere(np.isnan(preds_rescale.values)) #find nulls
        #delete make sure values are deleted from both
        y_test_rescale = np.delete(y_test_rescale.values, nans,axis=0)
        preds_rescale = np.delete(preds_rescale.values, nans, axis=0)
        index = np.delete(index, nans, axis=0)
        #turn back into dataframe in case next condition is also true
        preds_rescale = pd.DataFrame(preds_rescale, index=index) 
        y_test_rescale = pd.DataFrame(y_test_rescale, index=index)
        print('New Shape preds: ' + str(preds_rescale.shape))
        
    
    # if y_test_rescale.isnull().values.any()==True:
    #     print('Test data still have NaN values present. Deleting...')
    #     print('Current shape: ' + str(y_test_rescale.shape))
    #     nans = np.argwhere(np.isnan(y_test_rescale.values)) 
    #     #same as above
    #     y_test_rescale = np.delete(y_test_rescale.values, nans,axis=0)
    #     preds_rescale = np.delete(preds_rescale.values, nans, axis=0)
    #     index = np.delete(index, nans, axis=0)
    #     # make into DataFrame this time to guarantee the below return statement won't spit an error
    #     y_test_rescale = pd.DataFrame(y_test_rescale, index=index)
    #     preds_rescale = pd.DataFrame(preds_rescale, index=index)
    #     print('New Shape tests: ' + str(y_test_rescale.shape))
    
    print('test data new shape: ' + str(y_test_rescale.shape))
    print('prediction new shape: ' + str(preds_rescale.shape))
    return y_test_rescale, preds_rescale

def lstm_prep_no_pp(data_index, data, target_data, ntargets, ninputs, noutputs=1, show_progress=False):
    '''Prepares and reshapes data for use with an LSTM. Outputs features, targets,
    and the original data indices of your target values for visualization later. Requires that 
    the targets are the last N columns in your dataset.
    
    NOTE: The applies a moving window approach at intervals of the output steps, such that 
    you group the previous timesteps of inputs for your features (whatever length you choose), set the next 
    X timesteps of target values as outputs (again, whatever you want), and then move the window X (noutputs)
    timesteps in the future to repeat the process. Analogous to a cnn kernal with a stride equal to the output length. 
    I wrote this to automate and quickly change between varying input and output sequence lengths, 
    but wanted to avoid overlapping values typical in a moving window approach. 
    Having these non-overlapping values just makes plotting easier. 
    So far I have yet to see a need for more samples, which I understand is why the 
    moving window approach is typically implemented.'''
    
    #target_data = data[:,-ntargets:]
    target_data = target_data
    features = np.empty((ninputs, data.shape[1]), int)
    targets = np.empty((noutputs, ntargets), int)
    for i in trange(ninputs, (len(data)-noutputs), noutputs, desc=" data prep ", position=2, leave=False): 
        if show_progress==True:
            print('current index: ' + str(i))
        
        temp_feature_matrix = data[(i-ninputs):i,:]
        temp_target_matrix = target_data[(i):(i+noutputs)]
        
        features = np.vstack((features, temp_feature_matrix))
        targets = np.vstack((targets, temp_target_matrix))
    
    last_index = i+noutputs
    features = features.reshape((int(features.shape[0]/ninputs), ninputs, features.shape[1]))
    targets = targets.reshape(int(targets.shape[0]/noutputs), noutputs, targets.shape[1])
    
    target_indices = data_index[ninputs:last_index]
    
    return features[1:], targets[1:], target_indices

def lstm_prep_w_pp(data_index, data, ntargets, ninputs, noutputs=1, show_progress=False):
    '''Prepares and reshapes data for use with an LSTM. Outputs features, targets,
    and the original data indices of your target values for visualization later. Requires that 
    the targets are the last N columns in your dataset.
    
    NOTE: The applies a moving window approach at intervals of the output steps, such that 
    you group the previous timesteps of inputs for your features (whatever length you choose),set the next 
    X timesteps of target values as outputs (again, whatever you want), and then move the window X (noutputs)
    timesteps in the future to repeat the process. Analogous to a cnn kernal with a stride equal to the output length. 
    I wrote this to automate and quickly change between varying input and output sequence lengths, 
    but wanted to avoid overlapping values typical in a moving window approach. 
    Having these non-overlapping values just makes plotting easier. 
    So far I have yet to see a need for more samples, which I understand is why the 
    moving window approach is typicallyimplemented.'''
    
    target_data = data[:,-ntargets:]
    features = np.empty((ninputs, data.shape[1]), int)
    targets = np.empty((noutputs, ntargets), int)
    for i in trange(ninputs, (len(data)-noutputs), noutputs, desc=" data prep "):#, position=2, leave=True): 
        if show_progress==True:
            print('current index: ' + str(i))
        
        temp_feature_matrix = data[(i-ninputs):i,:]
        temp_target_matrix = target_data[(i):(i+noutputs)]
        
        features = np.vstack((features, temp_feature_matrix))
        targets = np.vstack((targets, temp_target_matrix))
    
    last_index = i+noutputs
    features = features.reshape((int(features.shape[0]/ninputs), ninputs, features.shape[1]))
    targets = targets.reshape(int(targets.shape[0]/noutputs), noutputs, targets.shape[1])
    
    target_indices = data_index[ninputs:last_index]
    
    return features[1:], targets[1:], target_indices

def rectify_cnn_data(predictions, targets, num_targets, noutputs):
    preds = np.empty((noutputs,num_targets), int)
    tests = np.empty((noutputs,num_targets), int)
    for row in range(0,predictions.shape[0]):
        pred_t = np.transpose(np.array(np.split(predictions[row],num_targets)))
        #print('preds: '+ str(pred_t))
        preds = np.vstack((preds, pred_t))

        test_t = np.transpose(np.array(np.split(targets[row],num_targets)))
        #print('tests: '+ str(test_t))
        tests = np.vstack((tests, test_t))
    #print(preds)
    return preds[noutputs:], tests[noutputs:]


# def mse_nan(y_true, y_pred):
#     masked_true = tf.where(tf.is_nan(y_true), tf.zeros_like(y_true), y_true)
#     masked_pred = tf.where(tf.is_nan(y_true), tf.zeros_like(y_true), y_pred)
#     return K.mean(K.square(masked_pred - masked_true), axis=-1)

# def mae_nan(y_true, y_pred):
#     masked_true = tf.where(tf.is_nan(y_true), tf.zeros_like(y_true), y_true)
#     masked_pred = tf.where(tf.is_nan(y_true), tf.zeros_like(y_true), y_pred)
#     return K.mean(abs(masked_pred - masked_true), axis=-1)

def mse_nan(y_true, y_pred):
    masked_true = tf.where(tf.is_nan(y_true), tf.zeros_like(y_true), y_true)
    masked_pred = tf.where(tf.is_nan(y_pred), tf.zeros_like(y_pred), y_pred)
    # m_true_targs = masked_true[-1]
    # m_pred_targs = masked_pred[-1]
    # return K.mean(K.square(m_pred_targs - m_true_targs), axis=-1)
    return K.mean(K.square(masked_pred - masked_true), axis=-1)

def mae_nan(y_true, y_pred):
    masked_true = tf.where(tf.is_nan(y_true), tf.zeros_like(y_true), y_true)
    masked_pred = tf.where(tf.is_nan(y_pred), tf.zeros_like(y_pred), y_pred)
    return K.mean(abs(masked_pred - masked_true), axis=-1)


def evaluate_forecasts(actual, predicted): #from Jason Brownlee's blog
    scores = list()
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        #print('mse: '+ str(mse))
        # calculate rmse
        rmse = np.sqrt(mse)
        # store
        scores.append(rmse)
    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col])**2
    score = np.sqrt(s / (actual.shape[0] * actual.shape[1]))
    
    print(scores)
    return score, scores

def threshold_rmse_eval(tests, preds, threshold):
    pred_pos_all = []
    y_pos_all = []
    rmse = []
    for i in range(preds.shape[1]):
        #grab individual cols
        data_slice_test = tests[:,i]
        data_slice_pred = preds[:,i]
        
        
        #This avoids a warning for the np.where query a couple of lines down...
        test_nans = np.where(np.isnan(data_slice_test)) #find nans and replace with dummy value. 
        tests[test_nans] = threshold*100
        
        #find all values greater than or equal to your threshold value
        pos_test = np.where(data_slice_test >= threshold)
        y_pos = data_slice_test[pos_test] #get those values from the test data
        pred_pos = data_slice_pred[pos_test] #get the equivalent values from the predictions
        
        #calculate mse, rmse
        mse = mean_squared_error(y_pos, pred_pos)
        rmse_val = np.sqrt(mse)
        
        #append all values to our respective lists
        rmse.append(rmse_val)
        y_pos_all.append(y_pos)
        pred_pos_all.append(pred_pos)
    
    print('per target rmse: ' + str(rmse))
    
    
    #make a list of rmse per datapoint
    diff_array = np.empty((1,tests.shape[1]),int)
    for row in range(tests.shape[0]):
        diffs = []
        for col in range(tests.shape[1]):
            #print(tests[row,col])
            #print(preds[row,col])
            diff = tests[row,col] - preds[row, col]
            diffs.append(diff)
        diffs = np.array(diffs)
        diff_array = np.vstack((diff_array,diffs))
    
    return y_pos_all, pred_pos_all, rmse, diff_array[1:]

def threshold_rmse_eval_rain(tests, preds, rain_data, threshold):
    rmse = []
    rain = rain_data.loc[tests.index]
    
    #find all values greater than or equal to your threshold value
    pos_test = np.where(rain >= threshold)
        
    for i in range(preds.shape[1]):
        #grab individual cols
        data_slice_test = tests.iloc[:,i]
        data_slice_pred = preds.iloc[:,i]
        
        
        #This avoids a warning for the np.where query a couple of lines down...
        test_nans = np.where(np.isnan(data_slice_test)) #find nans and replace with dummy value. 
        tests.iloc[test_nans] = 0
        
        pos_test_back = pos_test[0]-6
        pos_test_for = pos_test[0]+12
        
        # print(pos_test[0])
        # print(pos_test_back)
        # print(pos_test_for)
        
        y_pos = data_slice_test.iloc[pos_test_back[0]:pos_test_for[0]] #get those values from the test data
        pred_pos = data_slice_pred.iloc[pos_test_back[0]:pos_test_for[0]] #get the equivalent values from the predictions
        
        for kk in range(1,len(pos_test_back)):
            y_pos = pd.concat([y_pos,data_slice_test.iloc[pos_test_back[kk]:pos_test_for[kk]]])
            pred_pos = pd.concat([pred_pos,data_slice_pred.iloc[pos_test_back[kk]:pos_test_for[kk]]])
                
        y_pos = y_pos[~y_pos.index.duplicated(keep='first')]
        pred_pos = pred_pos[~pred_pos.index.duplicated(keep='first')]
        
        #calculate mse, rmse
        mse = mean_squared_error(y_pos, pred_pos)
        rmse_val = np.sqrt(mse)
        
        #append all values to our respective lists
        rmse.append(rmse_val)

    return rmse


def naive_forecast(test_data,backsteps,forward_steps):
    back_array = np.arange(0,backsteps+1)
    naive_preds = np.zeros(shape=(test_data.shape[0],test_data.shape[1]))
    for row in range(0,test_data.shape[0],forward_steps):
        #print(row)
        for col in range(test_data.shape[1]):
            if row in back_array:
                naive_preds[row,col] = test_data[row,col]
            else:
                naive_preds[row,col] = np.mean(test_data[(row-backsteps):row,col])
                for i in range(1,forward_steps):
                    row_index = row+i
                    if row_index == len(test_data):
                        break
                    else:
                        naive_preds[(row_index),col] = np.mean(naive_preds[(row_index-backsteps):row_index,col])

    return naive_preds

def add_max_rainfall(data, interval, rain_col, noise=False):
    '''takes times series data as input, and calculates the maximum and total rainfall values 
    for a fixed interval, based on the column index of your rainfall (rain_col).

    This function assumes you are using pandas, but can be modified for numpy arrays, which will perform faster
    for larger datasets'''
    rain_total = np.array(np.zeros(len(data)))
    rain_max = np.array(np.zeros(len(data)))
    
    if noise==True:
      for row in range(0,len(data),1):
        if row >= (len(data) - interval):
            rain_total[row] = (np.sum(data.iloc[row:len(data),rain_col]))*np.random.randint(0.75,1.25)
            rain_max[row] = (max(data.iloc[row:len(data),rain_col]))*np.random.randint(0.75,1.25)
            
        rain_total[row] = np.sum(data.iloc[row:row+interval,rain_col])*np.random.randint(0.75,1.25)
        rain_max[row] = max(data.iloc[row:row+interval,rain_col])*np.random.randint(0.75,1.25)

    else:
      for row in range(0,len(data),1):
          if row >= (len(data) - interval):
              rain_total[row] = np.sum(data.iloc[row:len(data),rain_col])
              rain_max[row] = np.nanmax(data.iloc[row:len(data),rain_col])
          
          rain_total[row] = np.sum(data.iloc[row:row+interval,rain_col])
          rain_max[row] = np.nanmax(data.iloc[row:row+interval,rain_col])

    data[str(interval)+'hr_total_rfall'] = rain_total
    data[str(interval)+'hr_max_rfall'] = rain_max
    
    return data

def OrdinalToDatetime(ordinal):
    plaindate = date.fromordinal(int(ordinal))
    date_time = datetime.combine(plaindate, datetime.min.time())
    dt = date_time + timedelta(days=ordinal-int(ordinal)) - timedelta(days=366) #as our serials start from 0000-01-00, instead of 0001-01-01
    return dt.replace(second=0, microsecond=0, minute=0, hour=dt.hour) +timedelta(hours=dt.minute//30)  #rounds to nearest hour, our data is in hourly increments


# Let's define a small function that predicts based on the trained encoder and decoder models
# Function also adapted from https://github.com/LukeTonin/keras-seq-2-seq-signal-prediction
def predict(x, encoder_predict_model, decoder_predict_model, num_steps_to_predict, 
            num_features_to_predict, batch_size=None):
    """Predict time series with encoder-decoder.
    
    Uses the encoder and decoder models previously trained to predict the next
    num_steps_to_predict values of the time series.
    
    Arguments
    ---------
    x: input time series of shape (batch_size, input_sequence_length, input_dimension).
    encoder_predict_model: The Keras encoder model.
    decoder_predict_model: The Keras decoder model.
    num_steps_to_predict: The number of steps in the future to predict
    num_features_to_predict: The number of features we want to predict per timestep
    
    Returns
    -------
    y_predicted: output time series for shape (batch_size, target_sequence_length,
        ouput_dimension)
    """
    y_predicted = []

    # Encode the values as a state vector
    enc_outputs_and_states = encoder_predict_model.predict(x, batch_size=batch, verbose=0)

    # The states must be a list
    if not isinstance(enc_outputs_and_states, list):
        enc_outputs_and_states = [enc_outputs_and_states]

    enc_outputs = enc_outputs_and_states[0]
    states = enc_outputs_and_states[1:]

    decoder_input = x[:,-1:,-num_features_to_predict:] # '-num_features_to_predict:' assumes the targets are the final columns

    for _ in range(num_steps_to_predict):
        outputs_and_states = decoder_predict_model.predict([x, decoder_input] + states, batch_size=batch, verbose=0)
        output = outputs_and_states[0]
        states = outputs_and_states[1:]

        # add predicted value
        y_predicted.append(output)
        #print(np.shape(y_predicted))


    return np.concatenate(y_predicted, axis=1)