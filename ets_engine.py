import os, sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score 

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from math import sqrt

def extract_param_count_hwes(config):
    return len(config['model'].keys()) + len(config['fit'].keys())

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def aic(n, mse, param_count):
    return n * np.log(mse) + 2 * param_count

def bic(n, mse, param_count):
    return n * np.log(mse) + param_count * np.log(n)
    
def calculate_errors(y_true, y_pred, param_count):
    # create a dictionary to store all of the metrics
    error_scores = {}
    pred_length = len(y_pred)
    try: 
        mse = mean_squared_error(y_true, y_pred)
    except ValueError:
        mse = 1e12
    try:
        error_scores['mae'] = mean_absolute_error(y_true, y_pred)
    except ValueError:
        error_scores['mae'] = 1e12
    error_scores['mape'] = mape(y_true, y_pred)
    error_scores['mse'] = mse
    error_scores['rmse'] = sqrt(mse)
    error_scores['aic'] = aic(pred_length, mse, param_count)
    error_scores['bic'] = bic(pred_length, mse, param_count)
    try:
        error_scores['explained_var'] = explained_variance_score(y_true, y_pred)
    except ValueError:
        error_scores['explained_var'] = -1e4
    try:
        error_scores['r2'] = r2_score(y_true, y_pred)
    except ValueError:
        error_scores['r2'] = -1e4
    
    return error_scores

def exp_smoothing_bayesian(train, test, selected_hp_values):
    output = {}
    exp_smoothing_model = ExponentialSmoothing(train,
                               trend=selected_hp_values['model']['trend'],
                               seasonal=selected_hp_values['model']['seasonal'],
                               damped_trend=selected_hp_values['model']['damped_trend'],
                               initialization_method=None
                                              )

    exp_fit = exp_smoothing_model.fit(smoothing_level=selected_hp_values['fit']['smoothing_level'],
                        smoothing_trend=selected_hp_values['fit']['smoothing_trend'],
                          smoothing_seasonal=selected_hp_values['fit']['smoothing_seasonal'],
                          damping_trend=selected_hp_values['fit']['damping_trend'],
                          method=selected_hp_values['fit']['method'],
                          remove_bias=selected_hp_values['fit']['remove_bias']
                         )

    forecast = exp_fit.predict(train.index[-1], test.index[-1])
    output['model'] = exp_fit
    output['forecast'] = forecast[1:]
    return output

def lasso_linear(params, *data):
    X_train, y_train = data

# params: alpha, tol
    params = {
              'alpha': params['alpha'], 
             'tol': params['tol']
             }
    # params = {'alpha': 0.3, 'tol': 0.001}
    linear_model = Lasso(**params)
    linear_model.fit(X_train, y_train)

    return linear_model


def calculate_errors_lasso(y_true, y_pred):
    # create a dictionary to store all of the metrics
    error_scores = {}
    # Here is populated dictionary with various metrics
    mse = mean_squared_error(y_true, y_pred)
    error_scores['mae'] = mean_absolute_error(y_true, y_pred)
    error_scores['mape'] = mape(y_true, y_pred)
    error_scores['mse'] = mse
    error_scores['rmse'] = sqrt(mse)
    error_scores['explained_var'] = explained_variance_score(y_true, y_pred)
    error_scores['r2'] = r2_score(y_true, y_pred)
    
    return error_scores
