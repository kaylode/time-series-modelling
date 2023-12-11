import matplotlib.pyplot as plt


from source.constants import TASK_COLUMNS
from source.visualize import visualize_ts
from source.models import (
    Tuner, validate, feature_engineering,
    fit_arima, fit_prophet, 
    predict_arima, predict_prophet,
    postprocess
)

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
parser.add_argument('--id', type=int, default=-1, required=False)

def fit_cv(
        df, time_column, value_column, 
        n_splits = 5, num_predictions=5, 
        tuner_config={},
        feature_method=None,
        method='arima', seasonal_lag=None, 
        visualize_freq='D', out_dir='data/results', best_key='mape'
    ):

        ## Split train and test
        train_index = list(range(df.shape[0] - num_predictions))
        test_index = list(range(df.shape[0] - num_predictions, df.shape[0]))
        train_df, test_df = df.iloc[train_index], df.iloc[test_index]

        # Fit model first time using cross validation to find best hyperparameters
        tuner_config['method'] = method
        tuner_config['storage'] = osp.join(out_dir, 'tuner', f'optuna.log')
        tuner_config['save_dir'] = osp.join(out_dir, 'tuner')
        os.makedirs(tuner_config['save_dir'], exist_ok=True)
        tuner = Tuner(**tuner_config)
        best_params = tuner.tune(train_df, params={
            'seasonal_lag': seasonal_lag,
            'method': method,
            'num_predictions': num_predictions,
            'n_splits': n_splits,
            'best_key': best_key,
            'time_column': time_column,
            'value_column': value_column,
            'fe_method': feature_method,
        })
        feature_method = best_params.get('fe_method', feature_method)

        # After finding best hyperparameters, fit the model again

        
        (
            norm_train, 
            engineered_train, 
            mu, 
            sigma
        ) = feature_engineering(
            train_df, 
            value_column=value_column,
            method=feature_method,
            seasonal_lag=seasonal_lag
        )

        # Visualize input to debug
        # visualize_ts(
        #     engineered_train.dropna().reset_index(), time_column, 0, 
        #     outpath=osp.join(out_dir, f'input.png'),
        #     freq=visualize_freq,
        #     figsize=(16,6),
        #     check_stationarity=True
        # )

        ## Fit best model (again) and make predictions on test set
        if method == 'arima':
            model = fit_arima(
                engineered_train, 
                best_params
            )
            pred_df = predict_arima(model, num_predictions)
        elif method == 'prophet':
            model = fit_prophet(
                engineered_train, 
                best_params
            )
            pred_df = predict_prophet(model, num_predictions)
        else:
            raise NotImplementedError

        # Match index (sometimes the index is not datetime, possibly a bug somewhere)
        # if pred_df.index.dtype != 'datetime64[ns]':
        #     pred_s = pred_df.shape[0]
        #     date_s = num_predictions + engineered_train.dropna().shape[0]
        #     missing = pred_s - date_s
        #     date_indexes = np.concatenate([engineered_train.dropna().index, test_df.index])
        
        #     if missing > 0:
        #         date_diff = date_indexes[-1] - date_indexes[-2]
        #         additional_dates = [date_indexes[-1] + date_diff * i for i in range(1, missing+1)]
        #         date_indexes = np.concatenate([date_indexes, additional_dates])
        #     pred_df.index = pd.to_datetime(date_indexes)
        #     pred_df = pred_df.asfreq(model.fittedvalues.index.freq)

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
            outpath=osp.join(out_dir, f'predict.png'),
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

        # Save predictions
        predictions = predictions.rename({'index': time_column}, axis=1)
        predictions.to_csv(osp.join(out_dir, 'predictions.csv'))

        return score

if __name__ == '__main__':

    args = parser.parse_args()
    DATA_DIR = args.data_dir
    OUT_DIR = args.out_dir
    task = 'AD&P'
    time_column = TASK_COLUMNS[task]['time_column']
    value_column = TASK_COLUMNS[task]['value_column']

    # load yaml file
    configs = yaml.load(open(args.config_file, 'r'), Loader=yaml.FullLoader)

    if args.id >= 0:
        filenames = [f'dataset_{args.id}.csv']
    else:
        filenames = sorted(os.listdir(DATA_DIR), key=lambda x: int(x.split('.')[0].split('_')[-1]))
    
    for filename in (pbar := tqdm(filenames)):
        pbar.set_description(f"Processing {filename}")
        filepath = osp.join(DATA_DIR, filename)
        file_prefix = filename.split('.')[0]
        config = configs[file_prefix]
        tuner_config = configs['tuner']
        tuner_config['study_name'] = file_prefix
        
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
            n_splits=tuner_config.pop('n_splits', 1),
            method=model_config['method'],
            num_predictions=model_config['num_predictions'],
            tuner_config=tuner_config,
            seasonal_lag=config.get('seasonality', None),
            visualize_freq=config['visualize_freq'],
            out_dir=out_dir,
            feature_method=model_config.get('fe_method', None),
        )

        with open(osp.join(out_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)
