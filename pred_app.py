import os, sys
import numpy as np
import pandas as pd 
import streamlit as st
import pickle
from ets_engine import exp_smoothing_bayesian, calculate_errors, extract_param_count_hwes, plot_predictions

st.set_page_config(
    page_title="Forecast",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title('Weekly Internet Sales Forecasting')
st.text_input('Input Date', '2019-01-01')

# with open(r"holt_winter_model.pickle", "rb") as input_file:
#    model = pickle.load(input_file)


def load_data():
    data = pd.read_csv(
        "Internet_sales_UK_preprocessed.csv",
        parse_dates=["date"],
        index_col=["date"],
    )

    int_col = list(data.select_dtypes("int").columns)
    float_col = list(data.select_dtypes("float").columns)
    data[int_col] = data[int_col].astype('int16')
    data[float_col] = data[float_col].astype('float32')

    data['Log_KPC4'] = np.log(data['KPC4'])
    data['Log_KPB8'] = np.log(data['KPB8'])

    kpc4_log_diff = data['Log_KPC4'].diff()
    kpc4_log_diff = kpc4_log_diff.dropna()
    kpb8_log_diff = data['Log_KPB8'].diff()
    kpb8_log_diff = kpb8_log_diff.dropna()
    return data

def splitting_data(data, split_date):
    train = data.loc[data.index < split_date]
    test = data.loc[data.index >= split_date]

    # the target variable
    y_train = train["KPC4"].copy()
    y_test = test["KPC4"].copy()

    # remove raw time series from predictors set
    X_train = train.drop(['KPC4','KPB8','KPB8_lag_1', 'KPB8_lag_3',
                        'KPB8_lag_6', 'KPB8_lag_12',
                        'KPB8_window_3_mean', 'KPB8_window_3_std',
                                'KPB8_window_6_mean', 'KPB8_window_6_std'], axis=1)
    X_test = test.drop(['KPC4','KPB8','KPB8_lag_1', 'KPB8_lag_3',
                        'KPB8_lag_6', 'KPB8_lag_12',
                        'KPB8_window_3_mean', 'KPB8_window_3_std',
                        'KPB8_window_6_mean', 'KPB8_window_6_std'], axis=1)

    return X_train, y_train, X_test, y_test


def main():
    np.random.seed(40)
    date = float(sys.argv[1]) if len(sys.argv) > 1 else "2019-01-01"
    data = load_data()

    X_train, y_train, X_test, y_test = splitting_data(data, date)
    best_result = {
        'model': {
            'trend': 'mul',
            'seasonal': 'add',
            'damped_trend': False
        },
        'fit': {
            'smoothing_level': 0.019176,
            'smoothing_trend': 0.03367,
            'smoothing_seasonal': 0.98565,
            'damping_trend': 0.061314,
            'method': "trust-constr",
            'remove_bias': False
        }
    }
    param_count = extract_param_count_hwes(best_result)
    model_results = exp_smoothing_bayesian(y_train, y_test, best_result)
    error_scores = calculate_errors(y_test, model_results['forecast'], param_count)
    
    st.write('The model performance of Forecasting with Exponential Smoothing.')

    st.write("mae", round(error_scores['mae'], 4))
    st.write("mape", round(error_scores['mape'], 4))
    st.write("mse", round(error_scores['mse'], 4))
    st.write("rmse", round(error_scores['rmse'], 4))
    st.write("aic", round(error_scores['aic'], 4))
    st.write("bic", round(error_scores['bic'], 4))
    st.write("explained_var", round(error_scores['explained_var'], 4))
    st.write("r2", round(error_scores['r2'], 4))

    X_test['Prediction'] = model_results['model'].forecast(len(y_test))

    # fig1 = plot_predictions(y_test, X_test['Prediction'], "Forecast Model", "Index Sales per Week", param_count)
    
    d = {'ground_truth': y_test, 'pred': X_test['Prediction']}
    chart_data = pd.DataFrame(data=d)
    st.line_chart(chart_data)

if __name__ == '__main__':
    main()