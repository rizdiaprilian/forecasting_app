import sys
import numpy as np
import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt
from ets_engine import exp_smoothing_bayesian, calculate_errors, extract_param_count_hwes
from ets_engine import lasso_linear, calculate_errors_lasso, arima_bayesian
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(
    page_title="Forecast",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title('Weekly Internet Sales Forecasting')

tab_titles = [
            "Tabular Viewing after Feature Engineering",
            "AutoCorrelation",
            "Forecasting with Lasso Regression", 
            "Forecasting with Exponential Smoothing",
            "Forecasting with ARIMA"
        ]
    
tabs = st.tabs(tab_titles)


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

def autocorr(series, lags):
    fig, axes = plt.subplots(2, 1, figsize=(8,12))
    fig = plot_acf(series, lags=lags, ax=axes[0])
    fig = plot_pacf(series, lags=lags, ax=axes[1])
    axes[0].set_xlabel('lags')
    axes[0].set_ylabel('correlation')
    axes[1].set_xlabel('lags')
    axes[1].set_ylabel('correlation')
    plt.show()

def main():
    np.random.seed(40)
    date = float(sys.argv[1]) if len(sys.argv) > 1 else "2019-01-01"
    data_preprocessed = load_data()
    data_preprocessed.round(4)
    data_rounded = data_preprocessed

    X_train, y_train, X_test, y_test = splitting_data(data_preprocessed, date)
    best_ets_result = {
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
    params = {
              'alpha': 0.12692, 
             'tol': 0.000277
             }

    best_arima_result = {
        'model': {
            'trend': "n", 
            'enforce_stationarity': False,
            'concentrate_scale': False
        },
        'fit': {
            'cov_type': "robust",
        }
    }

    ### Exponential Smoothing Forecasting
    param_ets_count = extract_param_count_hwes(best_ets_result)
    model_ets_results = exp_smoothing_bayesian(y_train, y_test, best_ets_result)
    error_ets_scores = calculate_errors(y_test, model_ets_results['forecast'], param_ets_count)

    ### ARIMA Forecasting
    endog=y_train
    param_count_arima = extract_param_count_hwes(best_arima_result)
    model_arima_results = arima_bayesian(endog, y_train, y_test, best_arima_result)
    error_arima_scores = calculate_errors(y_test, model_arima_results['forecast'], param_count_arima)

    ### Lasso Linear Forecasting
    linear_model = lasso_linear(params, X_train, y_train)
    y_pred_lasso = linear_model.predict(X_test)
    error_scores_lasso = calculate_errors_lasso(y_test, y_pred_lasso)

    with tabs[0]:
        st.subheader("Presenting Data after Feature Engineering")

        st.dataframe(data=data_rounded, use_container_width=False)

    with tabs[1]:
        st.subheader("Presenting Data AutoCorrelation")

        st.pyplot(fig=autocorr(y_train, 12), clear_figure=None)

    with tabs[2]:
        st.subheader("Model performance of Forecasting with Lasso Regression (Scikit-Learn).")

        st.write("mae", round(error_scores_lasso['mae'], 4))
        st.write("mape", round(error_scores_lasso['mape'], 4))
        st.write("mse", round(error_scores_lasso['mse'], 4))
        st.write("rmse", round(error_scores_lasso['rmse'], 4))
        st.write("explained_var", round(error_scores_lasso['explained_var'], 4))
        st.write("r2", round(error_scores_lasso['r2'], 4))

        d = {'ground_truth': y_test, 'pred': y_pred_lasso}
        chart_data_lasso = pd.DataFrame(data=d)
        st.line_chart(chart_data_lasso)

    with tabs[3]:
        st.write('Model performance of Forecasting with Exponential Smoothing (StatsModels).')

        st.write("mae", round(error_ets_scores['mae'], 4))
        st.write("mape", round(error_ets_scores['mape'], 4))
        st.write("mse", round(error_ets_scores['mse'], 4))
        st.write("rmse", round(error_ets_scores['rmse'], 4))
        st.write("aic", round(error_ets_scores['aic'], 4))
        st.write("bic", round(error_ets_scores['bic'], 4))
        st.write("explained_var", round(error_ets_scores['explained_var'], 4))
        st.write("r2", round(error_ets_scores['r2'], 4))

        X_test['Prediction_ETS'] = model_ets_results['model'].forecast(len(y_test))

        # fig1 = plot_predictions(y_test, X_test['Prediction'], "Forecast Model", "Index Sales per Week", param_count)
        
        d = {'ground_truth': y_test, 'pred': X_test['Prediction_ETS']}
        chart_data = pd.DataFrame(data=d)
        st.line_chart(chart_data)

    with tabs[4]:
        st.write('Model performance of Forecasting with ARIMA (StatsModels).')

        st.write("mae", round(error_arima_scores['mae'], 4))
        st.write("mape", round(error_arima_scores['mape'], 4))
        st.write("mse", round(error_arima_scores['mse'], 4))
        st.write("rmse", round(error_arima_scores['rmse'], 4))
        st.write("aic", round(error_arima_scores['aic'], 4))
        st.write("bic", round(error_arima_scores['bic'], 4))
        st.write("explained_var", round(error_arima_scores['explained_var'], 4))
        st.write("r2", round(error_arima_scores['r2'], 4))

        X_test['Prediction_ARIMA'] = model_arima_results['model'].forecast(len(y_test))

        # fig1 = plot_predictions(y_test, X_test['Prediction'], "Forecast Model", "Index Sales per Week", param_count)
        
        d = {'ground_truth': y_test, 'pred': X_test['Prediction_ARIMA']}
        chart_data = pd.DataFrame(data=d)
        st.line_chart(chart_data)

if __name__ == '__main__':
    main()
