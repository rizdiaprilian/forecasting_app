import os, sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
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

def plot_predictions(y_true, y_pred, time_series_name, value_name, param_count, plot_size=(10, 7)):
    # dictionary for currying
    validation_output = {} 
    
    # full error metrics suite as shown in listing 6.6
    error_values = calculate_errors(y_true, y_pred, param_count)
    
    # store all of the raw values of the errors
    validation_output['errors'] = error_values
    
    # create a string to populate a bounding box with on the graph
    text_str = '\n'.join((
        'mae = {:.3f}'.format(error_values['mae']),
        'mape = {:.3f}'.format(error_values['mape']),
        'mse = {:.3f}'.format(error_values['mse']),
        'rmse = {:.3f}'.format(error_values['rmse']),
        'aic = {:.3f}'.format(error_values['aic']),
        'bic = {:.3f}'.format(error_values['bic']),
        'explained var = {:.3f}'.format(error_values['explained_var']),
        'r squared = {:.3f}'.format(error_values['r2']),
    )) 
    with plt.style.context(style='seaborn'):
        fig, axes = plt.subplots(1, 1, figsize=plot_size)
        axes.plot(y_true, 'b-', label='Test data for {}'.format(time_series_name))
        axes.plot(y_pred, 'r-', label='Forecast data for {}'.format(time_series_name))
        axes.legend(loc='upper left')
        axes.set_title('Raw and Predicted data trend for {}'.format(time_series_name))
        axes.set_ylabel(value_name)
        axes.set_xlabel(y_true.index.name)
        
        # create an overlay bounding box so that all of our metrics are displayed on the plot
        props = dict(boxstyle='round', facecolor='oldlace', alpha=0.5)
        axes.text(0.05, 0.9, text_str, transform=axes.transAxes, fontsize=12, verticalalignment='top', bbox=props)
        plt.tight_layout()
        plt.show()
    return validation_output