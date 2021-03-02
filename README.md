# SNS-2021-SN15105271

Author: Yanwu Liu

Student ID: 15105271  

Module code: ELEC0088 Software for Network and Services Design 20/21 assignment 

email address: zczliue@ucl.ac.uk

## Table of Contents 
- [Introduction](#Introduction)
- [Files](#Files)
- [DataSplitting](#DataSplitting)
- [UserGuide](#UserGuide)
- [ErrorHandling](#ErrorHandling)
- [Prerequisites](#Prerequisites)


## Introduction
This is the GitHub repository for ELEC0088 SNS, containing a testbed for Python3 on Mac OS Catalina 10.15. A total of 3 tasks have been carried out in this assignment: 
- Recursively predict daily new confirmed cases in England area, from 24-Nov-2020 to 27-Dec-2020
- Recursively predict daily new hospital admissions in England area, from 30-Nov-2020 to 28-Dec-2020
- For each of the 2 objectives above, apply grid seach and walk forward validtion to compare and select the optimal model between time series forecasting methods: Long short-term memory (LSTM), Auto Regressive Integrated Moving Average (ARIMA), seasonal ARIMA (SARIMAX), simple exponential smoothing (SES), Holt-winters Exponential smoothing (HWES) and HWES damped trend.

## Files
This repository contains the following files:
- **Datasets (folder)**: 2 datasets are stored inside its subfolder England, containing daily new cases csv and daily new hospital admissions csv separately. Both are downloaded from [GOV.UK](https://coronavirus.data.gov.uk/). 
- main.py: main function file. Prints results only

- **common_model_functions.py**: contains functions that can be shared across 2 sections. This includde:
  - train test split
  - feature engineering
  - data preprocessing (i.e. acf, pacf, ADF test)
  - model formulation (LSTM, ARIMA, SARIMAX, SES, HWES, HWES damped)
  - grid search each model for optimal parameters based on minimum RMSE with train-validation set
  - recursively walk forward prediction (1 day ahead eachh time) on test set
  - calculate evaluation matrices: RMSE, MAE and R squared
  - plotting
  
- **sec1_dailycases.py**: function file for section 1: predicting daily new cases in England from 24-Nov-2020 to 27-Dec-2020. The following steps are carried out in sequence:
  - inspect the original dataset behaviour, perform stationary tests such as rolling mean, stddev, ADF test, ACF plot and PACF plot.
  - perform train-test split and feature engineering. Feature engineering is specially for LSTM. For univariate dataset, a window size of w is chosen to use the past w length of data as feature set.

  - grid search for optimal parameter combination for each time series model based on minimum validation RMSE

  - Perform prediction on test set
  - Calculate other evaluation matrices (MAE and R squared). Generate plots and tables.
  
- **sec2_dailyhealthcare.py**: function file for Section 2: predicting daily new hospital admissions in England from 30-Nov-2020 to 28-Dec-2020. Logic flow same as section 1. 


- Jupyter_report.pdf[/Jupyter_report.pdf]: a Jupyternotebook report printing all results for both section 1 and section 2.
- prerequisites.txt[/prerequisites.txt]: a text file summarising all prerequisites packages.


## DataSplitting

For both sections, 0.75 of dataset is used as training, 0.15 as validation and 0.10 as test set.

- Section 1: The dataset contains 333 samples from 30-Jan-2020 until 28-Dec-2020. Each sample represents the daily new confirmed cases. The test set represents predictions for 34 days, from 24-Nov-2020 to 27-Dec-2020.

- Section 2:  The dataset contains 285 samples from 19-March-2020 until 28-Dec-2020. Each sample represents the daily new patients admitted to hospital. The test set represents predictions for 29 days, from 30-Nov-2020 to 28-Dec-2020.


## UserGuide

- Install prerequisites 
- Download repository. Do not change the storage place of datasets. 
- Run python3 main.py

Each section roughly requires 3 hours to finish, so a total of 6 hours is required to print all results. 

The user may also change the parameter range in grid search under each section function file. 


## Prerequisites

- Mac OS Catalina 10.15
- Python 3.6.7
- pandas 1.1.4
- numpy 1.18.5
- csv 1.0
- json 2.0.9
- matplotlib 3.3.2
- cycler 0.10.0
- PIL 8.0.1
- tensorflow 2.3.1
- tensorflow.python.keras 2.4.0
- h5py 2.10.0
- scipy 1.5.4
- tensorboard 2.4.0
- sklearn 0.21.2
- statsmodels 0.12.1

Full list can be found in [prerequisites.txt](/prerequisites.txt)




