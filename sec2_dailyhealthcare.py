
# ELEC0088 SNS 
# author: Yanwu Liu 
# SN: 15105271


# This is the function file for Section 2: predicting daily new hospital admissions in England from 30-Nov-2020 to 28-Dec-2020
# The following sequence is same as Section 1. 


# 1. inspect the original dataset behaviour, perform stationary tests such as rolling mean, stddev, 
# ADF test, ACF plot and PACF plot.
#
# 2. perform train-test split and feature engineering. Feature engineering is specially for LSTM. For univariate 
# dataset, a window size of w is chosen to use the past w length of data as feature set.
#
# 3. grid search for optimal parameter combination for each time series model based on minimum validation RMSE
#
# 4. Perform prediction on test set
#
# 5. Calculate other evaluation matrices (MAE and R squared). Generate plots and tables. 


import common_model_functions as comm_func

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


#Folder that stores all csv
global basedir
basedir = './Datasets'

global SMALL_SIZE
SMALL_SIZE = 12

global MEDIUM_SIZE
MEDIUM_SIZE = 16

global BIGGER_SIZE
BIGGER_SIZE = 32

global savepath
savepath = 'sec2_dailyhealthcare_res'
if not os.path.isdir(savepath):
    os.makedirs(savepath)

num_epochs= 80
num_batch = 32

train_ratio = 0.75
val_ratio = 0.15
test_ratio = 0.10

#use past 14 days as history as a default value
default_time_lag = 7



#LSTM model
LSTM_model_para_dict = {'time_lag':[7, 14, 21],\
                        'num_LSTM_layer':[1,2],\
                        'learning_rate':[0.001], \
                        'beta_1':[0.9],\
                        'beta_2':[0.999],\
                        'epsilon':[1e-07], \
                        'num_epochs':[num_epochs],\
                        'num_batch':[num_batch]}

#ARIMA model
Arima_model_para_dict = {'p_values':list(range(0, 3)), 'd_values':list(range(0, 2)), 'q_values':list(range(0, 3))}


#SARIMAX
#the pdq values of SARIMAX will be using the best configurations combination returned by ARIMA model
seasonal_arima_paragrid = {'P_values':list(range(0, 3)),\
                           'D_values':list(range(0, 3)),\
                           'Q_values':list(range(0, 3)),\
                           'm_values':[7, 14],\
                           't_values':['n','c','t','ct']}



#Simple Exponential Smoothing 
SES_paragrid = {'initialization_method':[None,'estimated','heuristic','legacy-heuristic', 'known']}


#HWES model
HWES_paragrid = {'seasonal_periods': [7, 14],\
                 'trend':['additive',  'multiplicative'],\
                 'seasonal':['additive', 'multiplicative']}


#HWES with damped trend
#same as HWES_paragrid



#Support Vector Regression
SVR_paragrid = {'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                        'C': [1e-2, 1e-1, 1],
                        'gamma': [1e-3, 1e-2, 1e-1],
                        'degree':[8, 9, 10]}





def print_sec2_dailyhealthcare_results():

    print()
    print('-----------------------Print section 2: daily patients admitted to hospital:----------------------\n')

    print('-----------------------2.1 Data source inspection-----------------------\n')

    #This csv starts from 19 March until 27 Dec 2020, contains daily new patients admitted to hospital and cumulative
    #Source: https://coronavirus.data.gov.uk/
    #reverse dataframe so it stars from 30 Jan
    England_healthcare_csv_path = os.path.join(basedir, 'England','England_healthcare-Dec-28.csv')
    England_healthcare_28_Dec_2020 = pd.read_csv(England_healthcare_csv_path).iloc[::-1]

    #clear NAN values (if any)
    England_healthcare_28_Dec_2020.dropna(inplace=True)

    #show first a few data (from 19March)
    England_healthcare_28_Dec_2020.head()

    #show summary
    England_healthcare_28_Dec_2020.describe().transpose()

    #convert to date time object
    England_date_time = pd.to_datetime(England_healthcare_28_Dec_2020['date'], format='%d/%m/%Y')
    England_date_time_df = pd.DataFrame(England_date_time)

    #drop the strings such as 'areaType', 'areaCode' since all data are from England    
    #Series
    England_daily_healthcare_28_Dec_2020_series = England_healthcare_28_Dec_2020['newAdmissions']
    England_cum_healthcare_28_Dec_2020_series = England_healthcare_28_Dec_2020['cumAdmissions']

    #DataFrame
    England_daily_healthcare_28_Dec_2020_df = England_daily_healthcare_28_Dec_2020_series.to_frame()
    England_cum_healthcare_28_Dec_2020_df = England_cum_healthcare_28_Dec_2020_series.to_frame()


    #Plot Daily and Accumulative cases against time
    fig, (ax1, ax2) = plt.subplots(2,constrained_layout=True, figsize=(20,10))
    ax1.plot(England_date_time,England_daily_healthcare_28_Dec_2020_df, 'tab:blue')
    ax1.set_xlabel('Datetime', fontsize=MEDIUM_SIZE)
    ax1.set_ylabel('Daily new admissions', fontsize=MEDIUM_SIZE)
    ax1.set_title('Daily new hospital admissions in England since 19March2020 to 28Dec2020', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis="x", labelsize=MEDIUM_SIZE)
    ax1.tick_params(axis="y", labelsize=MEDIUM_SIZE)
    ax1.grid()

    ax2.plot(England_date_time,England_cum_healthcare_28_Dec_2020_df, 'tab:orange')
    ax2.set_xlabel('Datetime', fontsize=MEDIUM_SIZE)
    ax2.set_ylabel('Accumulated admissions', fontsize=MEDIUM_SIZE)
    ax2.set_title('Accumulated admissions in England since 19March2020 to 28Dec2020', fontsize=MEDIUM_SIZE)
    ax2.tick_params(axis="x", labelsize=MEDIUM_SIZE)
    ax2.tick_params(axis="y", labelsize=MEDIUM_SIZE)
    ax2.grid()
    plt.savefig(os.path.join(savepath,'Daily_new_healthcare_source_data' + '.png'))
    #plt.show()

    #Plot autocorrelation to check for a proper time lag
    comm_func.plot_autocorrelation(England_daily_healthcare_28_Dec_2020_series, savepath)

    #Plot rolling mean, rolling std dev and ADF to check stationarity
    comm_func.plot_stationarity(England_daily_healthcare_28_Dec_2020_series, default_time_lag, savepath)

    print()
    print('-----------------------1.2 train-test split-----------------------\n')
    #Note that in time-series forescasting the data should not be randomly shuffled

    #Series, dataframe
    Train_England_daily_healthcare_series, Train_datetime_df, \
    Val_England_daily_healthcare_series, Val_datetime_df, \
    Test_England_daily_healthcare_series, Test_datetime_df = comm_func.time_series_train_test_split(England_daily_healthcare_28_Dec_2020_series, England_date_time_df, train_ratio,val_ratio, test_ratio)



    #convert to pure values (list)
    Train_England_daily_healthcare_values = Train_England_daily_healthcare_series.values
    Val_England_daily_healthcare_values = Val_England_daily_healthcare_series.values
    Test_England_daily_healthcare_values = Test_England_daily_healthcare_series.values



    #normalisation
    x_scaler = MinMaxScaler(feature_range=(0,1))
    Train_England_daily_healthcare_values_norm = x_scaler.fit_transform(Train_England_daily_healthcare_values.reshape((-1,1)))
    Val_England_daily_healthcare_values_norm = x_scaler.transform(Val_England_daily_healthcare_values.reshape((-1,1)))
    Test_England_daily_healthcare_values_norm = x_scaler.transform(Test_England_daily_healthcare_values.reshape((-1,1)))


    print()
    print('-----------------------2.3 prediction on each model-----------------------\n')

    print('------------2.3.1 LSTM prediction on daily new healthcare------------\n')
    LSTM_test_cfg, LSTM_test_score, LSTM_test_predictions, LSTM_feature_engineering_dict, LSTM_train_val_history = \
                                comm_func.print_LSTM_results(Train_England_daily_healthcare_values_norm, \
                                                      Train_datetime_df,\
                                                      Val_England_daily_healthcare_values_norm , \
                                                      Val_datetime_df,\
                                                      Test_England_daily_healthcare_values_norm, \
                                                     Test_datetime_df,\
                                                      LSTM_model_para_dict, x_scaler, savepath)

    
    print()
    print('------------2.3.2 ARIMA prediction on daily new healthcare------------\n')
    best_ARIMA_cfg, best_ARIMA_rmse, ARIMA_test_rmse, ARIMA_test_predictions = comm_func.print_ARIMA_results(Train_England_daily_healthcare_values, \
                                                                                                   Val_England_daily_healthcare_values, \
                                                                                                  Test_England_daily_healthcare_values, \
                                                                                                   Arima_model_para_dict, Test_datetime_df, savepath)   


    print()     
    print('------------2.3.3 SARIMAX prediction on daily new healthcare------------\n')
    SARIMA_best_cfg, SARIMA_best_rmse, SARIMA_test_rmse, SARIMA_test_predictions = comm_func.print_SARIMAX_results(best_ARIMA_cfg, \
                                                                                     Train_England_daily_healthcare_values,\
                                                                                     Val_England_daily_healthcare_values,\
                                                                                     Test_England_daily_healthcare_values,\
                                                                                     seasonal_arima_paragrid,\
                                                                                     Test_datetime_df, savepath)

    
    print()
    print('-----------2.3.4 SES prediction on daily new healthcare----------------\n')
    SES_best_cfg, SES_best_rmse, SES_test_rmse, SES_test_predictions = comm_func.print_SES_results(Train_England_daily_healthcare_values, \
                                                                                         Val_England_daily_healthcare_values, \
                                                                                         Test_England_daily_healthcare_values, \
                                                                                         SES_paragrid, Test_datetime_df, savepath)

    print()
    print('----------2.3.5 HWES predictions on daily new healthcare----------------\n ')
    HWES_best_cfg, HWES_best_rmse, HWES_test_rmse, HWES_test_predictions = comm_func.print_HWES_results(Train_England_daily_healthcare_values, \
                                                                                         Val_England_daily_healthcare_values, \
                                                                                         Test_England_daily_healthcare_values, \
                                                                                         HWES_paragrid, Test_datetime_df, savepath)



    print()
    print('----------2.3.6 HWES with damping predictions on daily new healthcare----------------\n ')
    HWES_damped_best_cfg, HWES_damped_best_rmse, HWES_damped_test_rmse, HWES_test_damped_predictions = comm_func.print_HWES_damping_results(Train_England_daily_healthcare_values, \
                                                                                         Val_England_daily_healthcare_values, \
                                                                                         Test_England_daily_healthcare_values, \
                                                                                         HWES_paragrid, Test_datetime_df, savepath)


    
    print()
    print()
    print('*****************************************************************************')
    print('----------Summary of Section 2: Daily New healthcare admissions test results:--------------')

    print('LSTM test rmse: {} with optimal parameter set: {} \n'.format(LSTM_test_score, LSTM_test_cfg))

    print('ARIMA test rmse: {} with optimal parameter set: {}\n'.format(ARIMA_test_rmse, best_ARIMA_cfg))

    print('SARIMAX test rmse: {} with optimal parameter set: {}\n'.format(SARIMA_test_rmse, SARIMA_best_cfg))

    print('SES test rmse: {} with optimal parameter set: {}\n'.format(SES_test_rmse, SES_best_cfg))

    print('HWES test rmse: {} with optimal parameter set: {}\n'.format(HWES_test_rmse, HWES_best_cfg))

    print('HWES_damped test rmse: {} with optimal parameter set: {}\n'.format(HWES_damped_test_rmse, HWES_damped_best_cfg))



    #plot all models pedictions together
    all_model_pred_summary = {}
    all_model_pred_summary['LSTM'] = LSTM_test_predictions
    all_model_pred_summary['ARIMA'] = ARIMA_test_predictions
    all_model_pred_summary['SARIMAX'] = SARIMA_test_predictions
    all_model_pred_summary['SES'] = SES_test_predictions
    all_model_pred_summary['HWES'] = HWES_test_predictions
    all_model_pred_summary['HWES_damped'] = HWES_test_damped_predictions


    print()
    print('----print accuracy matrices:------\n')
    #calculates the test predictions regarding accuracy matrices: MAE, RMSE (same), R2
    comm_func.calculate_test_accuracy_matrices_summary(all_model_pred_summary, Test_England_daily_healthcare_values)

    #plot each prediction from models and true test values against time
    comm_func.plot_all_models_predictions_together(all_model_pred_summary, Test_England_daily_healthcare_values, Test_datetime_df, 'Daily new admissions', savepath)

    #plot each curve separately
    comm_func.subplot_predictions_summary(all_model_pred_summary, Test_England_daily_healthcare_values, Test_datetime_df, 'Daily new admissions', savepath)


    #print relative absolute error table
    #each sample RAE = ((y_pred - y_true)/y_true)
    print()
    print('----print relative absolute error table----\n')
    comm_func.print_relative_absolute_error_table(all_model_pred_summary, Test_England_daily_healthcare_values, Test_datetime_df)
    #plot relative absolute error against time
    comm_func.plot_relative_absolute_error_summary(all_model_pred_summary, Test_England_daily_healthcare_values, Test_datetime_df, 'relative absolute error', savepath)


    #plot relative accuracy
    #RA = y_pred/y_true
    print()
    print('----print relative accuracy table----\n')
    comm_func.print_relative_accuracy_table(all_model_pred_summary, Test_England_daily_healthcare_values, Test_datetime_df)
    #plot relative accuracy against time
    comm_func.plot_relative_accuracy_summary(all_model_pred_summary, Test_England_daily_healthcare_values, Test_datetime_df, 'relative accuracy', savepath)



    print()
    print('----End of Section 2: Daily new healthcare admissions ----\n')

