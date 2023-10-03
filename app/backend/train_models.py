#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train
"""

import pandas as pd
import os
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras import layers, metrics    
from tensorflow.keras.models import Model, load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KernelDensity
from joblib import dump, load
from matplotlib import pyplot


def set_subpaths(path):
    global DATAPATH
    global MODELPATH
    
    DATAPATH = path + '/data'
    MODELPATH = path + '/models'


#Data param
MINUTES_TREASHOLD_FOR_DEFINING_MISSING = 0
NUM_COL_NAMES = ['value_data'] #Set in Train.py
NR_OF_SERIES = len(NUM_COL_NAMES)
TRAIN_FRAC = 0.75
VAL_FRAC = 0.25

#Hyperparams
WINDOW_LENGTH = 100
BATCH_SIZE = 64
EPOCHS = 200



### Create dataloader that takes into account missing timepoints
# Standard shape is: (batch, time, features)
# Why this way? When a sensor has not reported as expected nothing is stored in our database.
#  Instead of creating a datset with missing which would be slow and blow up memory 
#  (and require alot of work since the timestamps that do exist is not exact)
#  here time windows with missing timepoints in them are discharded.
#  The index is passed along as an additional input but never used in the model.


class WindowGenerator:
    
    def __init__(self, df):
        """ 
        """
        self.df = df
        self.defect_windows = 0

    def __call__(self):

        for idx in range(len(self.df)):
            sub  = self.df[idx: (idx+WINDOW_LENGTH)]
            num_sub = tf.convert_to_tensor(sub[NUM_COL_NAMES])
            
            #Remove windows with at least one nan in it                  
            if np.any(np.isnan(num_sub.numpy())):
                continue
            
            #Remove end and beggining of serie
            if sub.shape[0] != WINDOW_LENGTH:
                continue
            
            #Removes windows with break in them
            if (any(sub[1:WINDOW_LENGTH]['breaks_before']) == False):

                if (any(sub[0:(WINDOW_LENGTH-1)]['breaks_after']) == False):
                    
                    yield (num_sub, tf.convert_to_tensor(idx + int(np.floor(WINDOW_LENGTH/2)))), num_sub
                else:
                    self.defect_windows += 1
                    continue
                
            else:
                self.defect_windows += 1
                continue
                
#ds = tf.data.Dataset.from_generator(WindowGenerator(df), output_types=((tf.float64, tf.int32), tf.float64), output_shapes = (([WINDOW_LENGTH, NR_OF_SERIES], []), [WINDOW_LENGTH, NR_OF_SERIES]))
#batches = ds.batch(BATCH_SIZE).prefetch(buffer_size=183)

#To check shapes etc:
# iterator = iter(ds_train)
# print(iterator.get_next())

# #To check defect windows:
# w1 = WindowGenerator(df)
# w1.defect_windows



### Models   
     
class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.InputLayer(input_shape=(WINDOW_LENGTH,NR_OF_SERIES)),
      tf.keras.layers.Flatten(),
      layers.Dense(int(WINDOW_LENGTH * NR_OF_SERIES/1.5), activation="relu"),
      layers.Dense(int(WINDOW_LENGTH * NR_OF_SERIES/2.5), activation="relu"),
      layers.Dense(3, activation="relu")])

    self.decoder = tf.keras.Sequential([
      layers.Dense(int(WINDOW_LENGTH * NR_OF_SERIES/2.5), activation="relu"),
      layers.Dense(int(WINDOW_LENGTH * NR_OF_SERIES/1.5), activation="relu"),
      layers.Dense(WINDOW_LENGTH * NR_OF_SERIES, activation="sigmoid"),
      tf.keras.layers.Reshape((WINDOW_LENGTH, NR_OF_SERIES), input_shape=(WINDOW_LENGTH * NR_OF_SERIES,))])

  def call(self, x):
    data, index = x #Trix to be able to pass index of window along, needed to calculate performance RE/ performance metrics
    encoded = self.encoder(data)
    decoded = self.decoder(encoded)
    return decoded


def train(path):
    
    #Load population of sensors
    pop = pd.read_csv(path + '/pop.csv')
    pop['report_frequency'] = np.nan
    pop['breaks_frequency'] = np.nan
    pop['mins'] = np.nan
    pop['maxs'] = np.nan
    pop['effective_windows'] = np.nan
    pop['cutoff_001'] = np.nan
    pop['train_time'] = np.nan
    
    ###Loop over datasets in pop
    for key, value in pop.iterrows():
        
        #Load data
        #_id = '64426ea461105318ce5fce2e' Have no saved model??? Training fails?
        _id = value['_id']
        now = time.time()
        print(_id, 'started!')
            
        df = pd.read_csv(DATAPATH + '/' + _id + '.csv')
        
        ### Fix time
        #Missin values are not denoted. Two indicator variables instead:
            #Break before indicates this is the first timepoint in a window, there are missing before this timepoint
            #Break after indicates this is the last timepoint in a window, there are missing after this timepoint
        df = df.astype({'time': 'datetime64[ns, Etc/UCT]'})
        df['time'] = df['time'].dt.tz_convert('Europe/Stockholm')
        df = df.sort_values('time')
        df.index = range(len(df))
        
        #Helper vars to identify breaks in windows
        df['time_delta'] =  df['time'] -  df['time'].shift(1)
        report_freq = df['time_delta'].median()
        df['breaks_before'] = df['time_delta'] > report_freq + pd.Timedelta(minutes=MINUTES_TREASHOLD_FOR_DEFINING_MISSING)
        df['breaks_after'] = df['breaks_before'].shift(-1)
        
        #Add metadata to pop
        pop_idx = int(np.where(pop['_id'] == _id)[0])
        pop.iloc[pop_idx, pop.columns.get_loc('report_frequency')] = report_freq
        pop.iloc[pop_idx, pop.columns.get_loc('breaks_frequency')] = np.sum(df['breaks_before']) / len(df)
        
        ### Normalize
        # MinMax standard method
        scaler = MinMaxScaler()
        scaler.fit(df[NUM_COL_NAMES])
        df[NUM_COL_NAMES] = scaler.transform(df[NUM_COL_NAMES])
        #postprocessed = scaler.inverse_transform(preprocessed)
        
        #Add normalization data to pop
        pop.iloc[pop_idx, pop.columns.get_loc('mins')] = str(scaler.data_min_)
        pop.iloc[pop_idx, pop.columns.get_loc('maxs')] = str(scaler.data_max_)
        
        #Split to train and val
        split_idx = int(TRAIN_FRAC*len(df))
        df_train = df[0:split_idx]
        df_val = df[split_idx:-1]
        
        #Load window datasets        
        ds_train = tf.data.Dataset.from_generator(WindowGenerator(df_train), output_types=((tf.float64, tf.int32), tf.float64), output_shapes = (([WINDOW_LENGTH, NR_OF_SERIES], []), [WINDOW_LENGTH, NR_OF_SERIES]))
        ds_train = ds_train.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE).cache()
        ds_val = tf.data.Dataset.from_generator(WindowGenerator(df_val), output_types=((tf.float64, tf.int32), tf.float64), output_shapes = (([WINDOW_LENGTH, NR_OF_SERIES], []), [WINDOW_LENGTH, NR_OF_SERIES]))
        ds_val = ds_val.batch(int(len(df_val)/10)).prefetch(tf.data.AUTOTUNE).cache()
        
        
        #Train
        
        autoencoder = AnomalyDetector()
        # Default for adam is 10e-3 = 0.001 
        autoencoder.compile(optimizer='adam', loss='mae', metrics=[metrics.MeanAbsoluteError()])
        
        if os.path.exists(MODELPATH+ '/' + _id +'_autoencoder'):
            
            print('Model training skipped, loading saved model!')
            
        else:
    
            callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',  patience=10, min_delta=1e-3),
                        tf.keras.callbacks.ModelCheckpoint(filepath=MODELPATH+ '/' + _id +'_autoencoder', monitor='val_loss', save_best_only=True)]
            history = autoencoder.fit(ds_train, epochs = EPOCHS, callbacks=[callback], validation_data = ds_val)
            autoencoder.summary(expand_nested=True)
                
            #Pic of training
            pyplot.plot(history.history['mean_absolute_error'])
            pyplot.plot(history.history['val_mean_absolute_error'])
            pyplot.title('model performance')
            pyplot.ylabel('mae')
            pyplot.xlabel('epoch')
            pyplot.legend(['train', 'val'], loc='upper left')
            #pyplot.show()
            pyplot.savefig(MODELPATH + '/'+  _id +'_training_pic.png')
            pyplot.close()
    
        # Load best model
        try:
            autoencoder = load_model(MODELPATH+ '/' + _id +'_autoencoder')
        except:
            print('Training faild!!!')
            continue
            
        
        ds = tf.data.Dataset.from_generator(WindowGenerator(df), output_types=((tf.float64, tf.int32), tf.float64), output_shapes = (([WINDOW_LENGTH, NR_OF_SERIES], []), [WINDOW_LENGTH, NR_OF_SERIES]))
        batches = ds.batch(10000).prefetch(tf.data.AUTOTUNE)
    
        error_batches = []
        index_batches = []
        for (x, middle_idx), y in batches:
            predictions = autoencoder.predict((x, middle_idx))
            errors = y - predictions
            error_batches.append(errors)
            index_batches.append(middle_idx)
        errors = tf.concat(error_batches, 0)
        indexes = tf.concat(index_batches, 0)
    
        
        #Add metadata to pop
        pop.iloc[pop_idx, pop.columns.get_loc('effective_windows')] = errors.shape[0]
    
        abs_errors = tf.abs(errors)
        #Train error over variables:
        #tf.reduce_mean(abs_errors, (0,1))
        
        #All ways of estimating pdf I know of have problems with tails,
        # trick here is to estimate on whole dist, but censor the tail by f.ex. "1 for above 1%" 
        
        #Later do for each variable collapsed in window....
        
        window_maes = tf.reduce_mean(abs_errors, (1,2))
        window_maes = np.round(window_maes.numpy(),2)*100
        
        #Tail prob estimated
        cutoff_001 = np.quantile(window_maes, q=(1-0.001), method='lower')
        
        #Add metadata to pop
        pop.iloc[pop_idx, pop.columns.get_loc('cutoff_001')] = cutoff_001
        
        #Past back to original df for visualization
        #Index to last timepoint
        normal_window_maes = window_maes[window_maes <= cutoff_001]
        normal_window_maes = normal_window_maes.reshape((len(normal_window_maes), 1))
        error_scaler = MinMaxScaler()
        error_scaler.fit(normal_window_maes)
        window_maes = window_maes.reshape((len(window_maes), 1))
        normalized_window_maes = error_scaler.transform(window_maes)
        
        indexes = indexes + 1 
        df['window_mae'] = np.nan
        df.iloc[indexes.numpy(), df.columns.get_loc('window_mae')] = normalized_window_maes
        df[NUM_COL_NAMES] = scaler.inverse_transform(df[NUM_COL_NAMES])
        df.to_csv(MODELPATH+ '/' + _id +'_df_for_visualization.csv')
        
        #Pdf estimated on raw window maes, kernel density do not seem to be handling values between 0-1 well
        model = KernelDensity(bandwidth=2, kernel='gaussian').fit(window_maes)
        
        dump(model, MODELPATH + '/' +  _id + '_pdf_model.joblib')
        #pdf = load(MODELPATH + '/pdf_' + _id + '.joblib')
        #pdf.score_samples(values)
        
        #Plot of density
        values = np.arange(window_maes.min(), window_maes.max(), 1)
        values = values.reshape((len(values), 1))
        probabilities = model.score_samples(values)
        probabilities = np.exp(probabilities)
        pyplot.hist(window_maes, bins=50, density=True)
        pyplot.plot(values[:], probabilities)
        pyplot.axvline(x = cutoff_001, color = 'r', label = '0.1 % cutoff')
        pyplot.legend()
        #pyplot.show()
        pyplot.savefig(MODELPATH + '/'+  _id +'_pdf.png')
        pyplot.close()
        
        #Save pop
        pop.iloc[pop_idx, pop.columns.get_loc('train_time')] = (time.time()-now)/60    
        print(_id, 'took', (time.time()-now)/60, 'minutes.')
        
    pop.to_csv(MODELPATH + '/pop_'+str(int(time.time()))+'.csv')

        
#pop2 = pd.read_csv(PROJECT_FOLDER + '/pop_1695287540.csv')   

