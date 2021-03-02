#!/usr/bin/env python
# coding: utf-8

# In[1]:

# ELEC0088 SNS 
# author: Yanwu Liu 
# SN: 15105271


#This file contains some common functions that can be shared across sections
# this contains:
# train test split
# feature engineering
# data preprocessing (i.e. acf, pacf, ADF test)
# model formulation (LSTM, ARIMA, SARIMAX, SES, HWES, HWES damped)
# grid search each model for optimal parameters based on minimum RMSE with train-validation set
# recursively walk forward prediction (1 day ahead eachh time) on test set
# After obtaining predictions for each model, calculate other evaluation matrices: MAE and R squared
# plotting


# In[2]:


import itertools 
import warnings
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import sklearn.metrics as metrics
from math import sqrt
from itertools import cycle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Activation
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.optimizers import Adam, SGD, RMSprop


from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from pandas.plotting import autocorrelation_plot
from pandas.tseries.offsets import DateOffset

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics import tsaplots
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
from statsmodels.tsa.holtwinters import SimpleExpSmoothing as SES
from statsmodels.tsa.statespace.sarimax import SARIMAX



#font size
global SMALL_SIZE
SMALL_SIZE = 12

global MEDIUM_SIZE
MEDIUM_SIZE = 16

global BIGGER_SIZE
BIGGER_SIZE = 32




#LSTM reshaping a sequence of time-series data
#use previous samples at t-1, t-2, ...t-lag as input features
#and use the value at t as label
def time_lag_generator(inputdata, lag):
    features, labels = [], []
    labels_index = []
    #at time instance t = val+lag (start from 0)
    for val in range(len(inputdata)- lag):
        
        #use previous t - l, t-2, ...t-lag as features
        features.append(inputdata[val : (val + lag)])
        #use the value at t as label
        labels.append(inputdata[val+lag])
        
        #store the index to refer to datetime
        labels_index.append(val+lag)
        
    return np.array(features), np.array(labels), np.array(labels_index)


# In[4]:


def get_acf_and_pacf(timeseries, savepath):
    
    
    #ACF -> to estimate MA terms
    plt.figure(figsize=(20,10))
    #tsaplots.plot_acf(timeseries, lags = np.arange(len(timeseries)))
    tsaplots.plot_acf(timeseries)
    #plt.title('Statsmodels Autocorrelation')
    plt.savefig(os.path.join(savepath,'Statsmodels Autocorrelation' + '.png'))
    #plt.show()
    
    #PACF -> to estimate AR terms
    plt.figure(figsize=(20,10))
    tsaplots.plot_pacf(timeseries)
    #plt.title('Statsmodels Partial Autocorrelation')
    plt.savefig(os.path.join(savepath,'Statsmodels Partial Autocorrelation' + '.png'))
    #plt.show()
    
    


# In[5]:


def plot_stationarity(timeseries, time_lag, savepath):
    
    # rolling statistics
    rolling_mean = timeseries.rolling(window=time_lag).mean()
    rolling_std = timeseries.rolling(window=time_lag).std()
    
    # rolling statistics plot
    plt.figure()
    plt.rcParams["figure.figsize"] = (20, 10)
    original = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.tick_params(axis="x", labelsize=MEDIUM_SIZE)
    plt.tick_params(axis="y", labelsize=MEDIUM_SIZE)
    plt.legend(loc='best', fontsize=MEDIUM_SIZE)
    plt.grid()
    plt.title('Rolling Mean & Standard Deviation', fontsize=MEDIUM_SIZE)
    plt.savefig(os.path.join(savepath,'Rolling_mean_and_std' + '.png'))
    #plt.show(block=False)
    
    # Dickey–Fuller test:
    result = adfuller(timeseries)
    print()
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))
    print()


# In[6]:


def plot_LSTM_learning_curve(LSTM_history, savepath):

    #plot learning curve
    plt.figure()
    plt.plot(LSTM_history.history['mean_squared_error'], linewidth=2)
    plt.plot(LSTM_history.history['val_mean_squared_error'], linewidth=2)
    plt.title('Learning curve: train vs validation error comparison', fontsize=BIGGER_SIZE)
    plt.tick_params(axis="x", labelsize=BIGGER_SIZE)
    plt.tick_params(axis="y", labelsize=BIGGER_SIZE)
    plt.xlabel('number of epochs', fontsize=BIGGER_SIZE)
    plt.ylabel('Loss/error', fontsize=BIGGER_SIZE)
    plt.legend(['train', 'val'], loc='best', fontsize=BIGGER_SIZE)
    plt.grid()
    plt.savefig(os.path.join(savepath,'LSTM_learning_curve.png'))
    #plt.show()
    
    


# In[7]:



#due to the reason that LSTM uses the number of timelag as previous features
#the number of actual predict test label or LSTM is shorter than other models
#to compensate for the time lag such that LSTM test set has the same length as other models
#we move the last timelag length of validation set to test set
def LSTM_val_test_timelag_conf(Val_df, Val_datetime, Test_df, Test_datetime, time_lag):
    
    
    #for features:
    #if input is dataframe
    if isinstance(Val_df, pd.DataFrame): 
        val_tail = Val_df.tail(time_lag)
        LSTM_Test_df = pd.concat([val_tail, Test_df], ignore_index=True)
        LSTM_Val_df = Val_df.drop(Val_df.tail(time_lag).index)
        
    
    #if input is list
    else:
        val_tail = Val_df[-time_lag:] 
        LSTM_Test_df = np.concatenate([val_tail, Test_df])
        LSTM_Val_df = Val_df[:-time_lag, :]
        
    
    
    #for datetime:
    if isinstance(Val_datetime, pd.DataFrame): 
        val_datetime_tail = Val_datetime.tail(time_lag)
        LSTM_Test_datetime = pd.concat([val_datetime_tail, Test_datetime], ignore_index=True)
        LSTM_Val_datetime = Val_datetime.drop(Val_datetime.tail(time_lag).index)
        
    else:
                
        val_datetime_tail = Val_datetime[-time_lag:] 
        LSTM_Test_datetime = np.concatenate([val_datetime_tail, Test_datetime])
        LSTM_Val_datetime = Val_datetime[:-time_lag, :]
        
        
    
    return LSTM_Val_df,LSTM_Val_datetime, LSTM_Test_df, LSTM_Test_datetime
    

    
def datetime_train_test_split(Train_datetime, Val_datetime, Test_datetime, train_label_index, val_label_index, test_label_index):
    
    #select datetime of train val, test labels 
    train_labels_datetime = Train_datetime.iloc[train_label_index, :]
    val_labels_datetime = Val_datetime.iloc[val_label_index, :]
    test_labels_datetime = Test_datetime.iloc[test_label_index, :]
    
    return train_labels_datetime, val_labels_datetime, test_labels_datetime


#feature engineering: Use time lag to create new feature and label set 
#at every instance t
#previous samples at [t-1, t-2, ...t-lag] are stored as input features
#the current value at t is the label 
#reshape into (num_samples, features, timestep) for feature set

def LSTM_feature_engineering(Train_df,Train_datetime, Val_df,Val_datetime, Test_df, Test_datetime, time_lag):

    
    #current shape is (num_samples, features, timestep)
    train_features_df, train_label_df, train_label_index= time_lag_generator(Train_df, time_lag)
    val_features_df, val_label_df, val_label_index= time_lag_generator(Val_df, time_lag)
    test_features_df, test_label_df, test_label_index= time_lag_generator(Test_df, time_lag)

    #reshape features into (num_samples, timestep, features)
    train_features_df= train_features_df.reshape(train_features_df.shape[0], 1, train_features_df.shape[1])
    val_features_df= val_features_df.reshape(val_features_df.shape[0], 1, val_features_df.shape[1])
    test_features_df= test_features_df.reshape(test_features_df.shape[0], 1, test_features_df.shape[1])


    #select datetime of train val, test labels 
    train_labels_datetime, val_labels_datetime, test_labels_datetime =     datetime_train_test_split(Train_datetime, Val_datetime, Test_datetime,                               train_label_index, val_label_index, test_label_index)


    print('Train history feature shape: {}, Train label shape: {}'.format(train_features_df.shape, len(train_label_df)))
    print('Val history feature shape: {}, Val label shape: {}'.format(val_features_df.shape, len(val_label_df)))
    print('Test history feature shape: {}, Test label shape: {}'.format(test_features_df.shape, len(test_label_df)))
    
    
    LSTM_feature_engineering_dict = {'train_features_df':train_features_df,                                     'train_label_df':train_label_df,                                      'train_labels_datetime':train_labels_datetime,                                    'val_features_df':val_features_df,                                     'val_label_df':val_label_df,                                      'val_labels_datetime':val_labels_datetime,                                    'test_features_df':test_features_df,                                    'test_label_df':test_label_df,                                     'test_labels_datetime':test_labels_datetime}
    
    return LSTM_feature_engineering_dict


# In[8]:



#design a user customised LSTM model
#The 1st layer is set to be GRU
#Then multiple layers of LSTM are stacked as the hidden layers 
#
def design_LSTM_model(LSTM_model_para):
    
    #scaling factor to control output neurons
    SCALING_FACTOR = 3
    SCALING_FACTOR_2 = 2
        
    TIME_LAG = LSTM_model_para['time_lag']
    NUM_LSTM_LAYER = LSTM_model_para['num_LSTM_layer']
    
    LEARNING_RATE  = LSTM_model_para['learning_rate']
    BETA_1 = LSTM_model_para['beta_1']
    BETA_2 = LSTM_model_para['beta_2']
    EPSILON = LSTM_model_para['epsilon']
    
    base_output_neurons = SCALING_FACTOR_2*TIME_LAG*pow(2, SCALING_FACTOR*NUM_LSTM_LAYER)
    dense_base_output_neurons = SCALING_FACTOR_2*TIME_LAG*pow(2, NUM_LSTM_LAYER)
    
    # Model building
    model = Sequential()
    model.add(GRU(base_output_neurons, input_shape=(1, TIME_LAG), return_sequences=True))
    model.add(Dropout(0.25))
    
    for i in range(NUM_LSTM_LAYER):
        
        #if it is last LSTM layer, return_sequences = false
        if i == NUM_LSTM_LAYER - 1:
            #the number of output neurons is descending at base of 2
            print('Adding last hidden LSTM layer '+str(i)+':')
            model.add(LSTM(int(base_output_neurons*pow(2, -i))))
            #the number of dropout neurons is ascending at base of 2
            model.add(Dropout(pow(0.25, i+1)))
            #add fully connected dense layer
            model.add(Dense(int(dense_base_output_neurons*pow(2, -i)), activation='relu'))
            #model.add(Dropout(pow(0.25, i+1)))
            
        else:    
            #the number of output neurons is descending at base of 2
            print('Adding hidden LSTM layer '+str(i)+':')
            model.add(LSTM(int(base_output_neurons*pow(2, -i)), return_sequences=True))
            #the number of dropout neurons is ascending at base of 2
            model.add(Dropout(pow(0.25, i+1)))
            #add fully connected dense layer
            model.add(Dense(int(dense_base_output_neurons*pow(2, -i)), activation='relu'))
            model.add(Dropout(pow(0.25, i+1)))
    
    
    #output layer
    model.add(Dense(1))    
    model.summary()
    
    model.compile(loss="mse",                  optimizer=Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON,),                  metrics=['mean_squared_error'])
    
    return model
    
    
    
    




# In[10]:


def LSTM_fit_and_walk_forward_validation(LSTM_feature_engineering_dict, LSTM_model_para):
    
    #unpack model parameters
    TIME_LAG = LSTM_model_para['time_lag']
    NUM_EPOCHS = LSTM_model_para['num_epochs']
    BATCH_SIZE = LSTM_model_para['num_batch']
    
    #unpack from returned dict
    train_features_df = LSTM_feature_engineering_dict['train_features_df']
    train_label_df = LSTM_feature_engineering_dict['train_label_df']
    train_labels_datetime = LSTM_feature_engineering_dict['train_labels_datetime']

    val_features_df = LSTM_feature_engineering_dict['val_features_df']
    val_label_df = LSTM_feature_engineering_dict['val_label_df']
    val_labels_datetime = LSTM_feature_engineering_dict['val_labels_datetime']

    test_features_df = LSTM_feature_engineering_dict['test_features_df']
    test_label_df = LSTM_feature_engineering_dict['test_label_df']
    test_labels_datetime = LSTM_feature_engineering_dict['test_labels_datetime']
    
    #design LSTM model
    LSTM_model = design_LSTM_model(LSTM_model_para)
    
    #--------Train and validation-------------
    LSTM_TrainVal_history = LSTM_model.fit(train_features_df, train_label_df, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,  \
                                            validation_data=(val_features_df, val_label_df), verbose=1)
    #give validation 
    
    
    #--------Predict on test set--------------
    #join train and val together as seed history
    TrainVal_feature_df  = np.concatenate([train_features_df, val_features_df])
    history = TrainVal_feature_df
    predictions = list()
    
    #step over for test set prediction
    for i in range(len(test_features_df)):
        
        #
        #since the history has a shape of (num_samples, prediction timestep, features)
        #then history[-1:, :] refers to the last component of the 1st axis,
        #which is the nearest past timelag length of features
        yhat = LSTM_model.predict(history[-1:, :])
        
        #append predictions, the shape of yhat is (1, 1), shown as [[value]]
        predictions.append(yhat[0])
        #update true value into history for next iteration
        #history.append(test_features_df[i])
        history = np.vstack((history,test_features_df[i][None]))


        
    return predictions, LSTM_TrainVal_history
    
    


# In[11]:


def fit_and_forecast_LSTM_model(Train_df,Train_datetime, Val_df,Val_datetime, Test_df, Test_datetime, LSTM_model_para, scaler):
    

    
    TIME_LAG = LSTM_model_para['time_lag']
    NUM_EPOCHS = LSTM_model_para['num_epochs']
    BATCH_SIZE = LSTM_model_para['num_batch']
    
    #adjust the validation and test set so that LSTM can produce desired length
    Val_df, Val_datetime, Test_df, Test_datetime = LSTM_val_test_timelag_conf(Val_df,Val_datetime, Test_df,Test_datetime, TIME_LAG)
        
    #feature engineering
    LSTM_feature_engineering_dict = LSTM_feature_engineering(Train_df,Train_datetime, Val_df,Val_datetime, Test_df, Test_datetime,TIME_LAG)

    
    #unpack from returned dict
    #test_features_df = LSTM_feature_engineering_dict['test_features_df']
    test_label_df = LSTM_feature_engineering_dict['test_label_df']
    test_labels_datetime = LSTM_feature_engineering_dict['test_labels_datetime']

    
    #prediction on test set
    raw_prediction, LSTM_train_val_history = LSTM_fit_and_walk_forward_validation(LSTM_feature_engineering_dict, LSTM_model_para)
    

    #inverse normalisation
    LSTM_prediction= scaler.inverse_transform(raw_prediction)
    true_test_label = scaler.inverse_transform(test_label_df)
    pred_test_label = LSTM_prediction
    
    # calculate out of sample error
    rmse = sqrt(mean_squared_error(true_test_label, pred_test_label))
    
    return rmse, LSTM_prediction,  LSTM_feature_engineering_dict, LSTM_train_val_history
    
    
    


# In[12]:


def evaluate_LSTM_models(Train_df,Train_datetime, Val_df,Val_datetime, Test_df, Test_datetime, LSTM_model_para_dict, scaler):
    
    best_score, best_cfg = float("inf"), None
    
    #for every combination of parameters
    keys, values = zip(*LSTM_model_para_dict.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    #for every combination of 
    for LSTM_model_para in permutations_dicts:

        print()
        print('----------------------Printing new LSTM model in para grid-------------------------')
        print('LSTM model: {}'.format(LSTM_model_para))
        rmse, prediction, feature_engineering_dict, train_val_history = fit_and_forecast_LSTM_model(Train_df,Train_datetime, Val_df,Val_datetime, Test_df, Test_datetime, LSTM_model_para, scaler)

        #update the best para set if a lower rmse found
        if rmse < best_score:
            best_score, best_cfg, best_prediction, best_feature_engineering_dict, best_train_val_history = rmse, LSTM_model_para, prediction, feature_engineering_dict, train_val_history
        print()
        print('--------------------------------------------------')
        print('LSTM model: {}, RMSE={}'.format(LSTM_model_para, rmse))
        print('--------------------------------------------------')
        print()

            
    
    print()
    print('------------------------***--------------------------')
    print('Best LSTM model:{}, RMSE={}'.format(best_cfg, best_score))

    return best_cfg, best_score, best_prediction, best_feature_engineering_dict, best_train_val_history
    
    
    


# In[13]:


#credit: https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/

# Evaluate an ARIMA model for a given order (p,d,q) and return RMSE
# We use walk forward (one step at a time) per prediction
def fit_and_forecast_arima_model(train_time_series, test_time_series, arima_order):
    # prepare training dataset
    
    #convert to series if input is dataframe
    if isinstance(train_time_series, pd.DataFrame): 
        train_time_series = train_time_series.value
        
    if isinstance(test_time_series, pd.DataFrame): 
        #convert to series
        test_time_series = test_time_series.value  
        
    train_time_series = train_time_series.astype('float32')
    test_time_series = test_time_series.astype('float32')
    
    train, test = train_time_series, test_time_series
    history = [x for x in train]
    # make predictions
    predictions = list()
    
    #one step forwadr at a time per prediction
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        #model_fit = model.fit(trend='nc', disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse, predictions

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_ARIMA_models(train_time_series, test_time_series, arima_para_dict):

    best_score, best_cfg = float("inf"), None
    
    p_values = arima_para_dict['p_values']
    d_values = arima_para_dict['d_values']
    q_values = arima_para_dict['q_values']
    
    
    #for each combination of p, d, q
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                
                #use try catch to solve non-invertible order combiination
                try:
                    rmse, predictions = fit_and_forecast_arima_model(train_time_series, test_time_series, order)
                    #print(rmse)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print()
                    print('--------------------------------------------------')
                    print('ARIMA model: {}, RMSE={}'.format(order, rmse))
                    print('--------------------------------------------------')
                    print()
                except:
                    continue
    print()
    print('------------------------***--------------------------')
    print('Best ARIMA model:{}, RMSE={}'.format(best_cfg, best_score))

    return best_cfg, best_score



    
    

    


# In[14]:


# Evaluate a seasonal ARIMA model for a given order (p,d,q) (P, D, Q)m and return RMSE
# use walk forward (one step at a time) per prediction
# This process is similar to arima model
# credit: https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/
def fit_and_forecast_seasonal_arima_model(train_time_series, test_time_series, sarima_order):
    
    order, sorder, trend = sarima_order
    
    #convert to series if input is dataframe
    if isinstance(train_time_series, pd.DataFrame): 
        train_time_series = train_time_series.value
        
    if isinstance(test_time_series, pd.DataFrame): 
        #convert to series
        test_time_series = test_time_series.value  
        
    train_time_series = train_time_series.astype('float32')
    test_time_series = test_time_series.astype('float32')
    
    train, test = train_time_series, test_time_series
    history = [x for x in train]
    # make predictions
    predictions = list()
    
    #one step forwadr at a time per prediction
    for t in range(len(test)):
        model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=0)
        #model_fit = model.fit(trend='nc', disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse, predictions



# evaluate combinations of p, d and q values for an ARIMA model (Grid search)
def evaluate_seasonal_ARIMA_models(train_time_series, test_time_series, seasonal_arima_para_dict):

    best_score, best_cfg = float("inf"), None
    
    p_values = seasonal_arima_para_dict['p_values']
    d_values = seasonal_arima_para_dict['d_values']
    q_values = seasonal_arima_para_dict['q_values']
    
    P_values = seasonal_arima_para_dict['P_values']
    D_values = seasonal_arima_para_dict['D_values']
    Q_values = seasonal_arima_para_dict['Q_values']
    m_values = seasonal_arima_para_dict['m_values']
    t_values = seasonal_arima_para_dict['t_values']
    
    #for each combination of p, d, q
    for p in p_values:
        for d in d_values:
            for q in q_values:
                for P in P_values:
                    for D in D_values:
                        for Q in Q_values:
                            for m in m_values:
                                for t in t_values:
                                    order = [(p,d,q), (P,D,Q,m), t]
                
                                    #use try catch to solve non-invertible order combiination
                                    try:
                                        rmse, predictions = fit_and_forecast_seasonal_arima_model(train_time_series, test_time_series, order)
                                        #print(rmse)
                                        if rmse < best_score:
                                            best_score, best_cfg = rmse, order
                                        print()
                                        print('--------------------------------------------------')
                                        print('SARIMAX model: {}, RMSE={}'.format(order, rmse))
                                        print('--------------------------------------------------')
                                        print()
                                    except:
                                        continue
    print()
    print('------------------------***--------------------------')
    print('Best SARIMAX model:{}, RMSE={}'.format(best_cfg, best_score))

    return best_cfg, best_score






# Evaluate a Simple Exponential smoothing and return RMSE
def fit_and_forecast_SES_model(train_time_series, test_time_series, SES_para):
    
    # prepare parameters
    INITIONALIZATION=SES_para['initialization_method']
    
    #convert to series if input is dataframe
    if isinstance(train_time_series, pd.DataFrame): 
        train_time_series = train_time_series.value
        
    if isinstance(test_time_series, pd.DataFrame): 
        #convert to series
        test_time_series = test_time_series.value  
        
    #drop any 0 values in training dataset
    train_time_series = [i for i in train_time_series if i != 0]
    test_time_series  = [i for i in test_time_series if i != 0]
    
    train, test = train_time_series, test_time_series
    history = [x for x in train]
    # make predictions
    predictions = list()
    
    for t in range(len(test)):
        model = SES(history, initialization_method = INITIONALIZATION)
        model_fit = model.fit(optimized=True, use_brute=True)
        yhat = model_fit.forecast(steps=1)
        predictions.append(yhat)
        history.append(test[t])
        
    
    # calculate out of sample error
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse, predictions



# evaluate combinations of HWES parameter grid
def evaluate_SES_models(train_time_series, test_time_series, SES_paragrid):
    
    INITIONALIZATION_list=SES_paragrid['initialization_method']

    best_score, best_cfg = float("inf"), None
    
    #for each combination in the paragrid:
    for ii in INITIONALIZATION_list:
                
        #form parameter combination
        SES_model_para = {'initialization_method': ii}
        try:
            rmse, predictions = fit_and_forecast_SES_model(train_time_series, test_time_series, SES_model_para)
            print(rmse)
            if rmse < best_score:
                best_score, best_cfg = rmse, SES_model_para
            print()
            print('--------------------------------------------------')
            print('SES model: {}, RMSE={}'.format(SES_model_para, rmse))
            print('--------------------------------------------------')
            print()
        except:
            continue
    print()
    print('------------------------***--------------------------')
    print('Best SES model:{}, RMSE={}'.format(best_cfg, best_score))

    return best_cfg, best_score
                





# Evaluate a Holt-winters Exponential smoothing and return RMSE
def fit_and_forecast_HWES_model(train_time_series, test_time_series, HWES_para):
    
    # prepare parameters
    SEASONAL_PERIODS=HWES_para['seasonal_periods']
    TREND=HWES_para['trend']
    SEASONAL=HWES_para['seasonal']
    
    #convert to series if input is dataframe
    if isinstance(train_time_series, pd.DataFrame): 
        train_time_series = train_time_series.value
        
    if isinstance(test_time_series, pd.DataFrame): 
        #convert to series
        test_time_series = test_time_series.value  
        
    #drop any 0 values in training dataset
    train_time_series = [i for i in train_time_series if i != 0]
    test_time_series  = [i for i in test_time_series if i != 0]
    
    train, test = train_time_series, test_time_series
    history = [x for x in train]
    # make predictions
    predictions = list()
    
    for t in range(len(test)):
        model = HWES(history, seasonal_periods = SEASONAL_PERIODS, trend = TREND, seasonal=SEASONAL)
        model_fit = model.fit(optimized=True, use_brute=True)
        yhat = model_fit.forecast(steps=1)
        predictions.append(yhat)
        history.append(test[t])
        
    
    # calculate out of sample error
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse, predictions



# evaluate combinations of HWES parameter grid
def evaluate_HWES_models(train_time_series, test_time_series, HWES_paragrid):
    
    SEASONAL_PERIODS_list=HWES_paragrid['seasonal_periods']
    TREND_list=HWES_paragrid['trend']
    SEASONAL_list=HWES_paragrid['seasonal']

    best_score, best_cfg = float("inf"), None
    
    #for each combination in the paragrid:
    for sp in SEASONAL_PERIODS_list:
        for tr in TREND_list:
            for sl in SEASONAL_list:
                
                #form parameter combination
                HWES_model_para = {'seasonal_periods': sp, 'trend':tr, 'seasonal': sl}
                try:
                    rmse, predictions = fit_and_forecast_HWES_model(train_time_series, test_time_series, HWES_model_para)
                    print(rmse)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, HWES_model_para
                    print()
                    print('--------------------------------------------------')
                    print('HWES model: {}, RMSE={}'.format(HWES_model_para, rmse))
                    print('--------------------------------------------------')
                    print()
                except:
                    continue
    print()
    print('------------------------***--------------------------')
    print('Best HWES model:{}, RMSE={}'.format(best_cfg, best_score))

    return best_cfg, best_score
                


# In[16]:


def plot_true_values_vs_prediction(true_test, pred_test,test_datetime, YLABEL, model_name, savepath):

    #plot Ture values vs prediction
    plt.figure()
    plt.rcParams["figure.figsize"] = (20, 10)
    plt.plot(test_datetime, true_test, 'b', marker = 'o', label = 'True values')
    plt.plot(test_datetime, pred_test, 'r', marker = 's', label = 'Predicted values')
    plt.ylabel(YLABEL,  fontsize=MEDIUM_SIZE)
    plt.xlabel('Datetime',  fontsize=MEDIUM_SIZE)
    plt.tick_params(axis="x", labelsize=MEDIUM_SIZE)
    plt.tick_params(axis="y", labelsize=MEDIUM_SIZE)
    plt.legend(loc="upper left", fontsize=MEDIUM_SIZE)
    plt.grid()
    plt.title("True {} vs Predictions comparison, {}".format(YLABEL, model_name), fontsize=MEDIUM_SIZE)
    plt.savefig(os.path.join(savepath,'True_{}_vs_Predictions_comparison_{}.png'.format(YLABEL, model_name)))
    #plt.show()


# In[17]:


def plot_all_models_predictions_together(all_model_predictions_summary, true_test, test_datetime, YLABEL, savepath):
    
    
    #colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'darkgreen', 'darkviolet'])
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    markers = cycle(('s', '+', 'x', 'd', '*')) 
    
    #plot Ture values vs prediction
    plt.figure()
    #plt.rcParams["figure.figsize"] = (20, 10)
    
    #plot true test
    plt.plot(test_datetime, true_test, '#7f7f7f', marker = 'o', label = 'True values', lw = 7)
    
    #plot predictions for each mdodel
    for model, COLOR, MARKER in zip(all_model_predictions_summary, colors, markers): 
        model_pred = all_model_predictions_summary[model]  
        plt.plot(test_datetime, model_pred, color = COLOR, marker = MARKER, label = model, lw = 2.5)
    
    
    plt.ylabel(YLABEL,  fontsize=BIGGER_SIZE)
    plt.xlabel('Datetime',  fontsize=MEDIUM_SIZE)
    plt.tick_params(axis="x", labelsize=MEDIUM_SIZE)
    plt.tick_params(axis="y", labelsize=BIGGER_SIZE)
    lgd = plt.legend(loc="upper left",  bbox_to_anchor=(1.05, 1), fontsize=BIGGER_SIZE)
    plt.grid()
    plt.title("True {} vs Predictions comparison, summary ".format(YLABEL), fontsize=BIGGER_SIZE)
    plt.savefig(os.path.join(savepath, '{}_all_models_summary.png'.format(YLABEL)), bbox_extra_artists=(lgd,), bbox_inches='tight')
    #plt.show()
    
    


# In[18]:


def plot_relative_absolute_error_summary(all_model_predictions_summary, true_test, test_datetime, YLABEL, savepath):
    
    #colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'darkgreen', 'darkviolet'])
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    markers = cycle(('s', '+', 'x', 'd', '*')) 
    
    true_test = true_test.reshape(true_test.shape[0],-1)
    
    plt.figure()
    plt.rcParams["figure.figsize"] = (20, 10)

    #plot predictions for each mdodel
    for model, COLOR, MARKER in zip(all_model_predictions_summary, colors, markers): 
        
        model_pred = np.array(all_model_predictions_summary[model])
        model_pred = model_pred.reshape(model_pred.shape[0],-1)
        abs_percentage_error = np.divide(np.absolute(np.subtract(true_test, model_pred)), true_test)

        
        plt.plot(test_datetime, abs_percentage_error, color = COLOR, marker = MARKER, label = model, lw = 2.5)
    
    
    plt.ylabel(YLABEL,  fontsize=BIGGER_SIZE)
    plt.xlabel('Datetime',  fontsize=MEDIUM_SIZE)
    plt.tick_params(axis="x", labelsize=MEDIUM_SIZE)
    plt.tick_params(axis="y", labelsize=BIGGER_SIZE)
    lgd = plt.legend(loc="upper left",  bbox_to_anchor=(1.05, 1), fontsize=BIGGER_SIZE)
    plt.grid()
    plt.title("Relative absolute error, summary", fontsize=BIGGER_SIZE)
    plt.savefig(os.path.join(savepath, '{}_RAE_summary.png'.format(YLABEL)), bbox_extra_artists=(lgd,), bbox_inches='tight')
    #plt.show()
    
def plot_relative_accuracy_summary(all_model_predictions_summary, true_test, test_datetime, YLABEL, savepath):
    
    #colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'darkgreen', 'darkviolet'])
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    markers = cycle(('s', '+', 'x', 'd', '*')) 
    
    true_test = true_test.reshape(true_test.shape[0],-1)
    
    plt.figure()
    plt.rcParams["figure.figsize"] = (20, 10)

    #plot predictions for each mdodel
    for model, COLOR, MARKER in zip(all_model_predictions_summary, colors, markers): 
        
        model_pred = np.array(all_model_predictions_summary[model])
        model_pred = model_pred.reshape(model_pred.shape[0],-1)
        percentage_accuracy = np.divide(model_pred, true_test) 

        
        plt.plot(test_datetime, percentage_accuracy, color = COLOR, marker = MARKER, label = model, lw = 2.5)
    
    
    plt.ylabel(YLABEL,  fontsize=BIGGER_SIZE)
    plt.xlabel('Datetime',  fontsize=MEDIUM_SIZE)
    plt.tick_params(axis="x", labelsize=MEDIUM_SIZE)
    plt.tick_params(axis="y", labelsize=BIGGER_SIZE)
    lgd = plt.legend(loc="upper left",  bbox_to_anchor=(1.05, 1), fontsize=BIGGER_SIZE)
    plt.grid()
    plt.title("Relative accuracy, summary", fontsize=BIGGER_SIZE)
    plt.savefig(os.path.join(savepath, '{}_RA_summary.png'.format(YLABEL)), bbox_extra_artists=(lgd,), bbox_inches='tight')
    #plt.show()    


# In[19]:


def plot_autocorrelation(time_series, savepath):
    #plot with pandas
    plt.figure()
    ax_acf = autocorrelation_plot(time_series)
    ax_acf.tick_params(axis='x',which='minor',direction='out',bottom=True,length=10)
    ax_acf.plot()
    plt.savefig(os.path.join(savepath,'pandas_ACF'+ '.png'))
    #plt.show()
    

    #plot with statsmodels
    if isinstance(time_series, pd.Series):
        LAGS = len(time_series)
    elif isinstance(time_series, pd.DataFrame):
        LAGS = len(time_series.index)
    else: #if list
        LAGS = len(time_series)
    LAGS_SPACE = 50

    plt.figure()
    tsaplots.plot_acf(time_series, lags = np.arange(LAGS - 4*LAGS_SPACE))
    plt.title('Statsmodels Autocorrelation')
    plt.savefig(os.path.join(savepath,'statsmodels_ACF'+ '.png'))
    #plt.show()
    
    #plot PACF    
    plt.figure()
    tsaplots.plot_pacf(time_series.squeeze(), lags = np.arange(LAGS_SPACE))
    plt.title('Statsmodels Partial Autocorrelation')
    plt.savefig(os.path.join(savepath,'statsmodels_PACF'+ '.png'))
    #plt.show()


# In[20]:


def plot_rolling_mean_and_stddev(time_series, time_lag, savepath):
    plt.figure()
    rolling_mean = time_series.rolling(window = time_lag).mean()
    rolling_std = time_series.rolling(window = time_lag).std()
    plt.plot(time_series, color = 'blue', label = 'Original')
    plt.plot(rolling_mean, color = 'red', label = 'Rolling Mean')
    plt.plot(rolling_std, color = 'black', label = 'Rolling Std')
    plt.legend(loc = 'best')
    plt.title('Rolling Mean & Rolling Standard Deviation')
    plt.savefig(os.path.join(savepath, 'Rolling_Mean_Rolling_Standard_Deviation.png'))
    plt.grid()
    #plt.show()


# In[21]:


def time_series_train_test_split(time_series, date_time, train_ratio,val_ratio, test_ratio):
    
    #Note that in time-series forescasting the data should not be randomly shuffled
    len_ts = len(time_series)

    #Series
    Train_series = time_series[0:int(len_ts*train_ratio)]
    Train_datetime = date_time[0:int(len_ts*train_ratio)]


    Val_series  = time_series[int(len_ts*train_ratio):int(len_ts*(train_ratio+val_ratio))]
    Val_datetime =  date_time[int(len_ts*train_ratio):int(len_ts*(train_ratio+val_ratio))]


    Test_series = time_series[int(len_ts*(train_ratio+val_ratio)):]                                                                                                                         
    Test_datetime = date_time[int(len_ts*(train_ratio+val_ratio)):]

    print()
    print('Raw Train length: {}'.format(len(Train_series)))
    print('Raw Val length: {}'.format(len(Val_series)))
    print('Raw Test length: {}'.format(len(Test_series)))
    print()
    
    return Train_series, Train_datetime, Val_series, Val_datetime, Test_series, Test_datetime

    


# Evaluate a Holt-winters Exponential smoothing and return RMSE
def fit_and_forecast_HWES_damped_model(train_time_series, test_time_series, HWES_para):
    
    # prepare parameters
    SEASONAL_PERIODS=HWES_para['seasonal_periods']
    TREND=HWES_para['trend']
    SEASONAL=HWES_para['seasonal']
    
    #convert to series if input is dataframe
    if isinstance(train_time_series, pd.DataFrame): 
        train_time_series = train_time_series.value
        
    if isinstance(test_time_series, pd.DataFrame): 
        #convert to series
        test_time_series = test_time_series.value  
        
    #drop any 0 values in training dataset
    train_time_series = [i for i in train_time_series if i != 0]
    test_time_series  = [i for i in test_time_series if i != 0]
    
    train, test = train_time_series, test_time_series
    history = [x for x in train]
    # make predictions
    predictions = list()
    
    for t in range(len(test)):
        model = HWES(history, seasonal_periods = SEASONAL_PERIODS, trend = TREND, seasonal=SEASONAL, damped_trend=True)
        model_fit = model.fit(optimized=True, use_brute=True)
        yhat = model_fit.forecast(steps=1)
        predictions.append(yhat)
        history.append(test[t])
        
    
    # calculate out of sample error
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse, predictions



# evaluate combinations of HWES parameter grid
def evaluate_HWES_damped_models(train_time_series, test_time_series, HWES_paragrid):
    
    SEASONAL_PERIODS_list=HWES_paragrid['seasonal_periods']
    TREND_list=HWES_paragrid['trend']
    SEASONAL_list=HWES_paragrid['seasonal']

    best_score, best_cfg = float("inf"), None
    
    #for each combination in the paragrid:
    for sp in SEASONAL_PERIODS_list:
        for tr in TREND_list:
            for sl in SEASONAL_list:
                
                #form parameter combination
                HWES_model_para = {'seasonal_periods': sp, 'trend':tr, 'seasonal': sl, 'damped_trend': 'True'}
                try:
                    rmse, predictions = fit_and_forecast_HWES_damped_model(train_time_series, test_time_series, HWES_model_para)
                    print(rmse)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, HWES_model_para
                    print()
                    print('--------------------------------------------------')
                    print('HWES model: {}, RMSE={}'.format(HWES_model_para, rmse))
                    print('--------------------------------------------------')
                    print()
                except:
                    continue
    print()
    print('------------------------***--------------------------')
    print('Best HWES model:{}, RMSE={}'.format(best_cfg, best_score))

    return best_cfg, best_score
               


# please note that the value does not indicate percentage, only relative accuracy
def print_relative_accuracy_table(all_model_predictions_summary, true_test, test_datetime):
    
    #formulate percentage accuracy table
    percentage_accuracy_summary = {}
    percentage_accuracy_summary['Test Date'] = (test_datetime.reset_index(drop=True)).values.flatten()
    #print(percentage_accuracy_summary['Test Date'].shape)
    
    true_test = true_test.reshape(true_test.shape[0],-1)

    for model in all_model_predictions_summary:
        model_pred = np.array(all_model_predictions_summary[model])
        model_pred = model_pred.reshape(model_pred.shape[0],-1)

        #no percentage
        percentage_accuracy = np.divide(model_pred, true_test) 

        percentage_accuracy_summary[model] = pd.Series(percentage_accuracy.flatten())
    
    percentage_accuracy_summary = pd.DataFrame(percentage_accuracy_summary) 
    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'expand_frame_repr', False):  
        print(percentage_accuracy_summary )
    


# please note that the value does not indicate percentage, only relative absolute error
def print_relative_absolute_error_table(all_model_predictions_summary, true_test, test_datetime):
    
    #formulate percentage accuracy table
    pae_summary = {}
    pae_summary['Test Date'] = (test_datetime.reset_index(drop=True)).values.flatten()
    #print(percentage_accuracy_summary['Test Date'].shape)
    
    true_test = true_test.reshape(true_test.shape[0],-1)

    for model in all_model_predictions_summary:
        model_pred = np.array(all_model_predictions_summary[model])
        model_pred = model_pred.reshape(model_pred.shape[0],-1)

        #no percentage
        abs_percentage_error = (np.divide(np.absolute(np.subtract(true_test, model_pred)), true_test))

        pae_summary[model] = pd.Series(abs_percentage_error.flatten())
    
    pae_summary = pd.DataFrame( pae_summary) 
    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'expand_frame_repr', False):  
        print(pae_summary)



def subplot_predictions_summary(all_model_predictions_summary, true_test, test_datetime, YLABEL, savepath):
    
    
    #colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'darkgreen', 'darkviolet'])
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    markers = cycle(('s', '+', 'x', 'd', '*')) 
    
    #plot Ture values vs prediction
    fig = plt.figure()
    fig.suptitle("True {} vs Predictions comparison, summary ".format(YLABEL), fontsize=BIGGER_SIZE)
    fig.text(0.5, 0.04, 'Datetime', ha='center', fontsize=MEDIUM_SIZE)
    fig.text(0.04, 0.5, YLABEL, va='center', rotation='vertical', fontsize=MEDIUM_SIZE)
    #plt.rcParams["figure.figsize"] = (20, 10)
    
    
    #plot predictions for each mdodel
    for model, COLOR, MARKER, ii in zip(all_model_predictions_summary, colors, markers, range(len(all_model_predictions_summary.keys()))): 
        model_pred = all_model_predictions_summary[model] 
        ax = fig.add_subplot(3, 2, ii+1)
        ax.plot(test_datetime, true_test, '#7f7f7f', marker = 'o', label = 'True values', lw = 4)
        ax.plot(test_datetime, model_pred, color = COLOR, marker = MARKER, label = model, lw = 2.5)
        ax.grid()
        ax.legend(loc="upper left")
        ax.set_xticks(ax.get_xticks()[::2])
    
    #fig.ylabel(YLABEL,  fontsize=BIGGER_SIZE)
    #fig.xlabel('Datetime',  fontsize=MEDIUM_SIZE)
    #fig.tick_params(axis="x", labelsize=MEDIUM_SIZE)
    #fig.tick_params(axis="y", labelsize=BIGGER_SIZE)

    plt.savefig(os.path.join(savepath, 'Daily_{}_all_models_summary_subplots.png'.format(YLABEL)))
    #plt.show()
    
    

#this calculates the test predictions regarding accuracy matrices
def calculate_test_accuracy_matrices_summary(all_model_predictions_summary, true_test):
    accuacy_matrices_summary = {}
    
    true_test = true_test.reshape(true_test.shape[0],-1)
    
    for model in all_model_predictions_summary:
        
        accuacy_matrices_summary[model] = {}
        model_pred = np.array(all_model_predictions_summary[model])
        model_pred = model_pred.reshape(model_pred.shape[0],-1)
        
        #MAE (Mean absolute error) 
        mae = metrics.mean_absolute_error(true_test, model_pred)
        #RMSE (Root Mean Squared Error) same as the train-val stage results
        rmse = sqrt(metrics.mean_squared_error(true_test, model_pred))
        #R squared (Coefficient of determination)
        r2 = metrics.r2_score(true_test, model_pred)
        
        accuacy_matrices_summary[model]['MAE'] = mae
        accuacy_matrices_summary[model]['RMSE'] = rmse
        accuacy_matrices_summary[model]['R2'] = r2
    
    
    accuacy_matrices_summary = pd.DataFrame(accuacy_matrices_summary) 
    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'expand_frame_repr', False):  
        print(accuacy_matrices_summary)
    



def print_LSTM_results(Train_norm, Train_datetime, Val_norm, Val_datetime, Test_norm, Test_datetime, LSTM_model_para_dict,x_scaler, savepath):
    
    #LSTM model with grid search:

    LSTM_test_cfg, LSTM_test_score, LSTM_test_predictions, LSTM_feature_engineering_dict, LSTM_train_val_history = \
                                    evaluate_LSTM_models(Train_norm, \
                                                      Train_datetime,\
                                                      Val_norm, \
                                                      Val_datetime,\
                                                      Test_norm, \
                                                      Test_datetime,\
                                                      LSTM_model_para_dict, x_scaler)



    #plot true test label and predicted values (with lowest rmse)
    plot_true_values_vs_prediction(x_scaler.inverse_transform(LSTM_feature_engineering_dict['test_label_df']), \
                               LSTM_test_predictions,\
                               LSTM_feature_engineering_dict['test_labels_datetime'], \
                               'Daily new cases', 'LSTM model', savepath)


    #plot learning curve for the best parameter combination
    plot_LSTM_learning_curve(LSTM_train_val_history, savepath)
    
    print('******')
    print('LSTM test rmse: {} with optimal parameter set: {}\n'.format(LSTM_test_score, LSTM_test_cfg))
    
    return  LSTM_test_cfg, LSTM_test_score, LSTM_test_predictions, LSTM_feature_engineering_dict, LSTM_train_val_history
   
    
    
    
 #ARIMA model
def print_ARIMA_results(Train_values, Val_values, Test_values, Arima_model_para_dict, Test_datetime,savepath):
    
    warnings.filterwarnings("ignore")

    #filter out the optimal p, d, q combination with lowest rmse with train and validation dataset
    best_ARIMA_cfg, best_ARIMA_rmse = evaluate_ARIMA_models(Train_values, Val_values, Arima_model_para_dict)

    #predict and compare with test set

    TrainVal_values  = np.concatenate([Train_values, Val_values])
    ARIMA_test_rmse, ARIMA_test_predictions = fit_and_forecast_arima_model(TrainVal_values , Test_values, best_ARIMA_cfg)


    #plot true test values against ARIMA predictions
    plot_true_values_vs_prediction(Test_values, ARIMA_test_predictions, Test_datetime, 'Daily new cases', 'ARIMA {} model'.format(best_ARIMA_cfg), savepath)

    print('******')
    print('ARIMA test rmse: {} with optimal parameter set: {}\n'.format(ARIMA_test_rmse, best_ARIMA_cfg))

    return best_ARIMA_cfg, best_ARIMA_rmse, ARIMA_test_rmse, ARIMA_test_predictions

    
def print_SARIMAX_results(best_ARIMA_cfg, Train_values, Val_values, Test_values, seasonal_arima_paragrid, Test_datetime, savepath):
    
    #update SARIMAX paragrid. 
    #To save computation time, the non-seasonal para (p, d, q) is directly assigned as the optimal para configurations 
    #found in the previous ARIMA test
    best_p_arima, best_d_arima, best_q_arima = best_ARIMA_cfg
    seasonal_arima_paragrid['p_values'] = [best_p_arima]
    seasonal_arima_paragrid['d_values'] = [best_d_arima]
    seasonal_arima_paragrid['q_values'] = [best_q_arima]


    #Train and test on validation set to evaluate the optimal configuration that gives the lowest validation rmse
    SARIMA_best_cfg, SARIMA_best_rmse = evaluate_seasonal_ARIMA_models(Train_values, Val_values, seasonal_arima_paragrid)

    #use both train and validation set for past history 
    #prediction is based on walk forward (1 step each time) method
    TrainVal_values  = np.concatenate([Train_values, Val_values])
    SARIMA_test_rmse, SARIMA_test_predictions = fit_and_forecast_seasonal_arima_model(TrainVal_values, Test_values, SARIMA_best_cfg)



    #plot true values and predictions
    plot_true_values_vs_prediction(Test_values, SARIMA_test_predictions, Test_datetime, 'Daily new cases', 'Seasonal ARIMA model:{}'.format(SARIMA_best_cfg), savepath)

    print('******')
    print('SARIMAX test rmse: {} with optimal parameter set: {}\n'.format(SARIMA_test_rmse, SARIMA_best_cfg))
    
    return SARIMA_best_cfg, SARIMA_best_rmse, SARIMA_test_rmse, SARIMA_test_predictions



#Simple Exponential Smoothing

def print_SES_results(Train_values, Val_values, Test_values, SES_paragrid, Test_datetime, savepath):
    
    warnings.filterwarnings("ignore")
    #filter out the optimal parameter combination with lowest rmse with train and validation dataset
    SES_best_cfg, SES_best_rmse = evaluate_SES_models(Train_values, Val_values, SES_paragrid)


    #use both train and validation set for past history 
    #prediction is based on walk forward (1 step each time) method
    TrainVal_values  = np.concatenate([Train_values, Val_values])
    SES_test_rmse, SES_test_predictions = fit_and_forecast_SES_model(TrainVal_values, Test_values, SES_best_cfg)



    #plot true vs HWES predictions
    plot_true_values_vs_prediction(Test_values, SES_test_predictions, Test_datetime, 'Daily new cases', 'Simple Exponential smoothing model:{}'.format(SES_best_cfg), savepath)


    print('******')
    print('SES test rmse: {} with optimal parameter set: {}'.format(SES_test_rmse, SES_best_cfg))


    return SES_best_cfg, SES_best_rmse, SES_test_rmse, SES_test_predictions




 #Holt-winters Exponential smoothing
def print_HWES_results(Train_values, Val_values, Test_values, HWES_paragrid, Test_datetime, savepath):

    warnings.filterwarnings("ignore")
    #filter out the optimal parameter combination with lowest rmse with train and validation dataset
    HWES_best_cfg, HWES_best_rmse = evaluate_HWES_models(Train_values, Val_values, HWES_paragrid)


    #use both train and validation set for past history 
    #prediction is based on walk forward (1 step each time) method
    TrainVal_values  = np.concatenate([Train_values, Val_values])
    HWES_test_rmse, HWES_test_predictions = fit_and_forecast_HWES_model(TrainVal_values, Test_values, HWES_best_cfg)



    #plot true vs HWES predictions
    plot_true_values_vs_prediction(Test_values, HWES_test_predictions, Test_datetime, 'Daily new cases', 'Holt-winters Exponential smoothing model:{}'.format(HWES_best_cfg), savepath)

    print('******')
    print('HWES test rmse: {} with optimal parameter set: {}'.format(HWES_test_rmse, HWES_best_cfg))
    
    return HWES_best_cfg, HWES_best_rmse, HWES_test_rmse, HWES_test_predictions





#HWES with damped trend

def print_HWES_damping_results(Train_values, Val_values, Test_values, HWES_paragrid, Test_datetime, savepath):

    warnings.filterwarnings("ignore")
    #filter out the optimal parameter combination with lowest rmse with train and validation dataset
    HWES_damped_best_cfg, HWES_damped_best_rmse = evaluate_HWES_damped_models(Train_values, Val_values, HWES_paragrid)


    #use both train and validation set for past history 
    #prediction is based on walk forward (1 step each time) method
    TrainVal_values  = np.concatenate([Train_values, Val_values])
    HWES_damped_test_rmse, HWES_test_damped_predictions = fit_and_forecast_HWES_damped_model(TrainVal_values, Test_values, HWES_damped_best_cfg)


    #plot true vs HWES predictions
    plot_true_values_vs_prediction(Test_values, HWES_test_damped_predictions, Test_datetime, 'Daily new cases', 'Holt-winters Exponential smoothing model:{}'.format(HWES_damped_best_cfg), savepath)



    print('******')
    print('HWES_damped test rmse: {} with optimal parameter set: {}'.format(HWES_damped_test_rmse, HWES_damped_best_cfg))

    return HWES_damped_best_cfg, HWES_damped_best_rmse, HWES_damped_test_rmse, HWES_test_damped_predictions




    