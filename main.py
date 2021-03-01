

# ELEC0088 SNS 
# author: Yanwu Liu 
# SN: 15105271

# This is the main file for the assignment.
# It only prints out the results. 
# Please see the function files for each section for details.


#import function files
import common_model_functions as comm_func

#import Section 1: daily new cases
import sec1_dailycases as sec1
#import Section 2: daily new hospital admissions
import sec2_dailyhealthcare as sec2



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





def main():
    #print section 1 results: predicting daily new confirmed cases in England from 24-Nov-2020 to 27-Dec-2020
    sec1.print_sec1_dailycases_results()

    #print section 2 results: predicting daily new hospital admissions in England from 30-Nov-2020 to 28-Dec-2020
    sec2.print_sec2_dailyhealthcare_results()




if __name__ == '__main__':
    main()
