from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
import matplotlib.pyplot as plt

from prophet import Prophet

from source.constants import TASK_COLUMNS
from sklearn.model_selection import TimeSeriesSplit
from source.visualize import visualize_ts

import os
import os.path as osp
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--out_dir', type=str, default='data/results')
parser.add_argument('--config_file', type=str, default='data/AD&P.yaml')

def feature_engineering(df, value_column, method='diff', mu=None, sigma=None, seasonal_lag=None):

    assert method in ['none', 'diff', 'diff2', 'seasonal_diff', 'norm'], f"Invalid method: {method}"

    def normalize_ts(df, value_column, mu=None, sigma=None):
        if mu is None:
            mu = np.mean(df[value_column])
        if sigma is None:
            sigma = np.std(df[value_column])
    
        norm_values = df.apply(lambda x: (x[value_column] - mu) / sigma, axis=1)
        return norm_values, mu, sigma

    norm_values, mu, sigma = normalize_ts(df, value_column, mu, sigma)
    if method == 'diff':
        engineered_values = norm_values.diff()
    elif method == 'diff2':
        engineered_values = norm_values.diff()
        engineered_values = engineered_values.diff()
    elif method == 'seasonal_diff':
        engineered_values = norm_values.diff(periods=seasonal_lag)
    elif method == 'none':
        engineered_values = df[value_column]
        mu, sigma = None, None
    elif method == 'norm':
        engineered_values = norm_values
    else:
        raise NotImplementedError

    return norm_values, engineered_values, mu, sigma

def fit_arima(df, model_configs={}, use_sarimax=False):

    # Fit ARIMA model
    if use_sarimax:
        model = SARIMAX(df.dropna(), **model_configs)
    else:
        model = ARIMA(df.dropna(), **model_configs)
    fit_model = model.fit()    
    # print(fit_model.summary())
    return fit_model


def fit_auto_arima(df, model_configs={}):
    model_config.update({
        'trace': True,  #logs 
        'error_action': 'warn',  #shows errors ('ignore' silences these)
        'suppress_warnings': False 
    })
    model = pm.auto_arima(
        df.dropna(), 
        **model_configs
    )

    print(model.summary())
    return model
    
    
def predict_auto_arima(model, num_predictions=5):
    fitted, confint = model.predict(n_periods=num_predictions, return_conf_int=True)
    lower_bound = pd.Series(confint[:, 0], index=fitted.index)
    upper_bound = pd.Series(confint[:, 1], index=fitted.index)
    pred_df = pd.DataFrame({
        'predicted_mean': fitted,
        'lower y': lower_bound,
        'upper y': upper_bound
    })
    
    return pred_df


def predict_arima(fit_model, num_predictions=5):
    fitted_values = fit_model.fittedvalues
    predictions = fit_model.get_prediction(0, fitted_values.shape[0] + num_predictions)
    yhat = predictions.predicted_mean
    yhat_conf_int = predictions.conf_int(alpha=0.05)
    pred_df = pd.merge(yhat.reset_index(), yhat_conf_int.reset_index(), on='index')
    pred_df = pred_df.set_index('index')
    return pred_df

def fit_prophet(df, model_configs={}):
    model = Prophet(**model_configs)
    model.fit(df)
    return model

def predict_prophet(model, num_predictions=5):
    future = model.make_future_dataframe(periods=num_predictions)
    predictions = model.predict(future)
    result = predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    result = result.set_index('ds')
    result = result.rename(columns={'yhat': 'predicted_mean', 'yhat_lower': 'lower y', 'yhat_upper': 'upper y'})
    
    return result.tail(num_predictions)

def postprocess(
        pred_df, 
        last_indexes=None, last_values=None, 
        sigma=None, mu=None, 
        cumsum=None, cumsum_periods=1
    ):

    # Reverse differencing
    if cumsum: #assuming df is already cumsumed
        if last_indexes is not None and last_values is not None:
            for col in pred_df.columns:
                for last_index, last_value in zip(last_indexes, last_values):
                    if last_index in pred_df.index:
                        continue
                    pred_df.at[last_index, col] = last_value
        pred_df = pred_df.sort_index()
        # Customized cumsum
        # index is datetime
        for index in pred_df.index[cumsum_periods:]:
            for col in pred_df.columns:
                pred_df.at[index, col] = (
                    pred_df.at[index, col] 
                    + pred_df.at[index-df.index.freq*cumsum_periods, 'predicted_mean']
                )
        if last_indexes is not None:
            pred_df = pred_df.drop(index=last_indexes)

    # Reverse normalization
    if sigma is not None and mu is not None:
        pred_df = pred_df * sigma + mu

    return pred_df


def validate(actual, forecast):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr

    return({
        'mape':mape, 'me':me, 'mae': mae, 
        'mpe': mpe, 'rmse':rmse,  
        'corr':corr, 
    })


def fit_cv(
        df, time_column, value_column, 
        n_splits = 5, num_predictions=5, model_configs={}, 
        method='arima', seasonal_lag=None, feature_method='diff',
        visualize_freq='D', out_dir='data/results'
    ):
    
    def get_mean_std(scores):
        return (
            np.mean(scores), \
            np.std(scores)
        )
    
    if n_splits > 1:
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=num_predictions)
        indexes = tscv.split(df)
    else:
        train_index = list(range(df.shape[0] - num_predictions))
        test_index = list(range(df.shape[0] - num_predictions, df.shape[0]))
        indexes = [(train_index, test_index)]
    scores = []

    # Split the time series using TimeSeriesSplit
    for fold_id, (train_index, test_index) in enumerate(indexes):
        train_df, test_df = df.iloc[train_index], df.iloc[test_index]
        (
            norm_train, 
            engineered_train, 
            mu, 
            sigma
        ) = feature_engineering(
            train_df, value_column, method=feature_method, seasonal_lag=seasonal_lag
        )

        # Visualize input to debug
        # visualize_ts(
        #     engineered_train.dropna().reset_index(), time_column, 0, 
        #     outpath=osp.join(out_dir, f'debug_{fold_id}.png'),
        #     freq=visualize_freq,
        #     figsize=(16,6),
        #     check_stationarity=True
        # )

        if method in ['sarimax', 'arima']:
            # Fit model
            if method == 'sarimax':
                use_sarimax = True
            else:
                use_sarimax = False
            model = fit_arima(engineered_train, model_configs, use_sarimax=use_sarimax)
            # Make predictions
            pred_df = predict_arima(model, num_predictions=num_predictions)

        elif method == 'auto_arima':
            model = fit_auto_arima(engineered_train, model_configs)
            pred_df = predict_auto_arima(model, num_predictions=num_predictions)
        
        elif method == 'prophet':
            engineered_train = engineered_train.reset_index()
            engineered_train = engineered_train.rename(
                columns={time_column: 'ds', engineered_train.columns[-1]: 'y'})
            # Fit model
            model = fit_prophet(engineered_train, model_configs)
            # Make predictions
            pred_df = predict_prophet(model, num_predictions=num_predictions)
        else:
            raise NotImplementedError

        # Match index (sometimes the index is not datetime, possibly a bug somewhere)
        if pred_df.index.dtype != 'datetime64[ns]':
            pred_s = pred_df.shape[0]
            date_s = num_predictions + engineered_train.dropna().shape[0]
            missing = pred_s - date_s
            date_indexes = np.concatenate([engineered_train.dropna().index, test_df.index])
        
            if missing > 0:
                date_diff = date_indexes[-1] - date_indexes[-2]
                additional_dates = [date_indexes[-1] + date_diff * i for i in range(1, missing+1)]
                date_indexes = np.concatenate([date_indexes, additional_dates])
            pred_df.index = pd.to_datetime(date_indexes)

        # Visualize forecast
        # Postprocess prediction
        no = seasonal_lag if feature_method == 'seasonal_diff' else 1
        last_indexes = norm_train.index[:no]
        last_values = norm_train.iloc[:no]
        predictions = postprocess(
            pred_df, sigma=sigma, mu=mu,
            last_indexes=last_indexes,
            last_values=last_values,
            cumsum=feature_method in ['diff', 'diff2', 'seasonal_diff'],
            cumsum_periods=seasonal_lag if feature_method == 'seasonal_diff' else 1,
        )


        # Clear memory of matplotlib
        plt.cla()
        plt.clf()
        plt.close('all')
        
        # Visualize predictions
        visualize_ts(
            train_df.reset_index(), time_column, value_column, 
            outpath=osp.join(out_dir, f'predict_{fold_id}.png'),
            freq=visualize_freq,
            targets=test_df.reset_index(),
            predictions=predictions['predicted_mean'],
            lower_bound=predictions['lower y'],
            upper_bound=predictions['upper y'],
            figsize=(16,4),
            # zoom=10
        )

        # Clear memory of matplotlib
        plt.cla()
        plt.clf()
        plt.close('all')

        # Evaluate the model
        score = validate(
            test_df[value_column].values, 
            predictions['predicted_mean'].loc[test_df.index].values
        )
        scores.append(score)

    avg_score = {}
    for metric in scores[0].keys():
        avg_score[metric] = get_mean_std([score[metric] for score in scores])

    return avg_score

if __name__ == '__main__':

    args = parser.parse_args()
    DATA_DIR = args.data_dir
    OUT_DIR = args.out_dir
    task = 'AD&P'
    time_column = TASK_COLUMNS[task]['time_column']
    value_column = TASK_COLUMNS[task]['value_column']

    # load yaml file
    configs = yaml.load(open(args.config_file, 'r'), Loader=yaml.FullLoader)

    filenames = sorted(os.listdir(DATA_DIR), key=lambda x: int(x.split('.')[0].split('_')[-1]))
    for filename in (pbar := tqdm(filenames)):
        pbar.set_description(f"Processing {filename}")
        filepath = osp.join(DATA_DIR, filename)
        file_prefix = filename.split('.')[0]
        config = configs[file_prefix]
        
        if 'forecast' not in config:
            continue
        
        model_config = config['forecast']
        df = pd.read_csv(filepath)
        df[time_column] = pd.to_datetime(df[time_column])

        df = df.sort_values(by=time_column)
        df = df.set_index(time_column)
        df = df.resample(config['freq']).mean()
        
        out_dir = osp.join(OUT_DIR, 'forecast', file_prefix)
        os.makedirs(out_dir, exist_ok=True)

        metrics = fit_cv(
            df, time_column, value_column,
            n_splits=model_config['n_splits'],
            method=model_config['method'],
            feature_method=model_config['feature_method'],
            num_predictions=model_config['num_predictions'],
            model_configs=model_config['model_configs'],
            seasonal_lag=config.get('seasonality', None),
            visualize_freq=config['visualize_freq'],
            out_dir=out_dir
        )

        with open(osp.join(out_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)
