from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from sklearn.metrics import mean_squared_error


def feature_engineering(df):
    mu, sigma = normalize_ts(df)

    df['kpi_value_diff'] = df['kpi_value_norm'].diff()
    df['kpi_value_diff2'] = df['kpi_value_diff'].diff()

    seasonal_lag = 96 # lag for exactly 24 hours
    df['kpi_value_seasonal_diff'] = df['kpi_value_norm'].diff(periods=seasonal_lag)


def perform_adf_test(series):
    result = adfuller(series)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])


def search_arima(df):
    perform_adf_test(df['kpi_value_seasonal_diff'].dropna())
    plot_pacf(df['kpi_value_seasonal_diff'])
    plot_acf(df['kpi_value_seasonal_diff'])



def fit_arima(df):
    p = 1
    d = 1
    q = 1

    order = (p, d, q)  # Replace p, d, q with appropriate values
    model = ARIMA(df['kpi_value_diff'].dropna(), order=order)
    fit_model = model.fit()
    return fit_model

def predict_arima(df, fit_model):
    predictions = fit_model.get_forecast(5)
    yhat = predictions.predicted_mean
    yhat_conf_int = predictions.conf_int(alpha=0.05)
    pred_df = pd.merge(yhat.reset_index(), yhat_conf_int.reset_index(), on='index')
    pred_df = pred_df.set_index('index')
    pred_df

    yhat[2852] = df['kpi_value_norm'].iloc[2852]
    yhat = yhat.sort_index()
    yhat_cumsum = yhat.cumsum()
    yhat_ori = yhat_cumsum * sigma + mu


    mse = mean_squared_error(test['kpi_value'], predictions)
    print(f'Mean Squared Error: {mse}')


def fit_cv(df):
    # Assume df is your DataFrame with time series data
    # Each row should have a timestamp, kpi_value, and an identifier for the time series (e.g., 'id')

    # Initialize TimeSeriesSplit with the number of splits (folds) you want
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Placeholder for metrics
    mse_scores = []


    # Iterate through each time series
    for id, group_df in df.groupby('id'):
        # Split the time series using TimeSeriesSplit
        for train_index, test_index in tscv.split(group_df):
            train, test = group_df.iloc[train_index], group_df.iloc[test_index]

            # Extract features and target variables
            X_train, y_train = train[['timestamp']], train['kpi_value']
            X_test, y_test = test[['timestamp']], test['kpi_value']

            import pdb; pdb.set_trace()

            # Replace the following with your chosen model
            # model = ExponentialSmoothing(y_train)
            # model_fit = model.fit()

            # # Make predictions
            # y_pred = model_fit.predict(start=test.index[0], end=test.index[-1])

            # # Evaluate the model
            # mse = mean_squared_error(y_test, y_pred)
            # mse_scores.append(mse)

    # Calculate the average MSE across all folds
    mean_mse = np.mean(mse_scores)
    std_mse = np.std(mse_scores)
    return mean_mse, std_mse