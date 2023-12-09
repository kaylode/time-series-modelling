import pandas as pd
import os
import os.path as osp
import matplotlib.pyplot as plt
from tqdm import tqdm
from statsmodels.tsa.seasonal import STL
import json
from source.visualize import visualize_ts, visualize_stl
from source.constants import TASK_COLUMNS
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--out_dir', type=str, default='data/anomalies')
parser.add_argument('--config_file', type=str, default='data/AD&P.yaml')

def detect_anomalies_rolling_mean(
        df, time_column, value_column,
        window_size=10, threshold=2.0, 
        out_dir=None, visualize_freq='D'
    ):

    # Compute rolling mean
    df['rolling_mean'] = df[value_column].rolling(window=window_size).mean()
    diff_rolling_mean = df['rolling_mean'].diff()

    diff_rolling_mean_mean = diff_rolling_mean.mean()
    diff_rolling_mean_std = diff_rolling_mean.std()
    lower = diff_rolling_mean_mean - threshold*diff_rolling_mean_std
    upper = diff_rolling_mean_mean + threshold*diff_rolling_mean_std

    # Plot rolling mean
    plt.figure(figsize=(16,4))
    plt.plot(df[time_column], df['rolling_mean'], label='Rolling mean')
    plt.plot(df[time_column], diff_rolling_mean, label='Rolling mean residuals')
    plt.xticks(rotation=90)
    plt.fill_between(
        [df[time_column].min(), df[time_column].max()], 
        lower, upper, color='g', alpha=0.25, 
        linestyle='--', linewidth=2
    )
    plt.legend()
    plt.savefig(osp.join(out_dir, 'rolling_mean.png'), bbox_inches='tight')

    # Mark anomalies based on threshold (you can adjust this threshold as needed)
    anomalies = df[(diff_rolling_mean < lower) | (diff_rolling_mean > upper)]

    visualize_ts(
        df, time_column, value_column, 
        outpath=osp.join(out_dir, 'anomalies_rolling_mean.png'),
        freq=visualize_freq,
        anomalies=anomalies,
        figsize=(16,4)
    )
    return anomalies

def detect_anomalies_rolling_std(
        df, time_column, value_column,
        window_size, threshold=2.0, 
        out_dir=None, visualize_freq='D'
    ):
    # Compute rolling standard deviation
    df['rolling_std'] = df[value_column].rolling(window=window_size).std()
    diff_rolling_std = df['rolling_std'].diff()

    diff_rolling_std_mean = diff_rolling_std.mean()
    diff_rolling_std_std = diff_rolling_std.std()
    lower = diff_rolling_std_mean - threshold*diff_rolling_std_std
    upper = diff_rolling_std_mean + threshold*diff_rolling_std_std

    # Plot rolling std
    plt.figure(figsize=(16,4))
    plt.plot(df[time_column], df['rolling_std'], label='Rolling std')
    plt.plot(df[time_column], diff_rolling_std, label='Rolling std residuals')
    plt.xticks(rotation=90)
    plt.fill_between(
        [df[time_column].min(), df[time_column].max()], 
        lower, upper, color='g', alpha=0.25, 
        linestyle='--', linewidth=2
    )
    plt.legend()
    plt.savefig(osp.join(out_dir, 'rolling_std.png'), bbox_inches='tight')

    # Mark anomalies based on threshold (you can adjust this threshold as needed)
    anomalies = df[(diff_rolling_std < lower) | (diff_rolling_std > upper)]
    visualize_ts(
        df, time_column, value_column, 
        outpath=osp.join(out_dir, 'anomalies_rolling_std.png'),
        freq=visualize_freq,
        anomalies=anomalies,
        figsize=(16,4)
    )
    return anomalies

def detect_anomalies_stl(
        df, time_column, value_column, 
        period=None,
        visualize_freq='D',
        threshold=3,
        out_dir:str=None
    ):

    df = df.sort_values(by=time_column)

    # Decompose into trend, seasonal, and residual
    stl = STL(df[value_column], period=period)
    result = stl.fit()
    seasonal, trend, resid = result.seasonal, result.trend, result.resid
    visualize_stl(
        df[value_column], 
        trend, seasonal, resid, 
        out_dir=out_dir,
        figsize=(16,16)
    )

    # Plot the original series and the estimated trend
    estimated = trend + seasonal
    plt.figure(figsize=(16,4))
    plt.plot(df[value_column])
    plt.plot(estimated)
    plt.savefig(osp.join(out_dir, 'estimated.png'), bbox_inches='tight')
    plt.close()

    # Detect anaomalies based on residuals
    resid_mu = resid.mean()
    resid_dev = resid.std()
    lower = resid_mu - threshold*resid_dev
    upper = resid_mu + threshold*resid_dev
    plt.figure(figsize=(16,4))
    plt.plot(resid)
    plt.fill_between(
        [resid.index.min(), resid.index.max()], 
        lower, upper, color='g', alpha=0.25, 
        linestyle='--', linewidth=2
    )
    plt.savefig(osp.join(out_dir, 'residual.png'), bbox_inches='tight')

    # Visualize anomalies
    anomalies = df[(resid < lower) | (resid > upper)]
    visualize_ts(
        df, time_column, value_column, 
        outpath=osp.join(out_dir, 'anomalies_stl.png'),
        freq=visualize_freq,
        anomalies=anomalies,
        figsize=(16,4)
    )

    return anomalies

def impute_anomalies(df, value_column, anomalies):
    # Impute new values for anomalies, using mean of the previous and next values

    new_df = df.copy()
    anomalies_idx = anomalies.index
    for idx in anomalies_idx:
        if 0 < idx < len(new_df) - 1:  # Check if index is within the valid range
            left_value = new_df.loc[idx - 1, value_column]
            right_value = new_df.loc[idx + 1, value_column]
            mean_value = (left_value + right_value) / 2.0
            new_df.at[idx, value_column] = mean_value
    return new_df
   
if __name__ == '__main__':

    args = parser.parse_args()
    DATA_DIR = args.data_dir
    OUT_DIR = args.out_dir
    task = 'AD&P'
    time_column = TASK_COLUMNS[task]['time_column']
    value_column = TASK_COLUMNS[task]['value_column']

    # load yaml file
    configs = yaml.load(open(args.config_file, 'r'), Loader=yaml.FullLoader)

    filenames = sorted(os.listdir(DATA_DIR))
    for filename in tqdm(filenames):
        filepath = osp.join(DATA_DIR, filename)
        file_prefix = filename.split('.')[0]
        config = configs[file_prefix]
        df = pd.read_csv(filepath)
        df[time_column] = pd.to_datetime(df[time_column])

        out_dir = osp.join(OUT_DIR, file_prefix)
        os.makedirs(out_dir, exist_ok=True)

        if 'anomalies' not in config:
            continue

        if config['anomalies']['method'] == 'stl':
            anomalies = detect_anomalies_stl(
                df, time_column, value_column,
                period=config['seasonality'],
                visualize_freq=config['visualize_freq'],
                threshold=config['anomalies']['threshold'],
                out_dir=out_dir
            )
        elif config['anomalies']['method'] == 'rolling_mean':
            anomalies = detect_anomalies_rolling_mean(
                df, time_column, value_column,
                window_size=config['anomalies']['window_size'], 
                threshold=config['anomalies']['threshold'], 
                visualize_freq=config['visualize_freq'],
                out_dir=out_dir
            )
        elif config['anomalies']['method'] == 'rolling_std':
            anomalies = detect_anomalies_rolling_std(
                df, time_column, value_column,
                window_size=config['anomalies']['window_size'], 
                threshold=config['anomalies']['threshold'], 
                visualize_freq=config['visualize_freq'],
                out_dir=out_dir    
            )
        else:
            raise NotImplementedError
        
        # If data has labels, try evaluating the anomaly detection method
        if 'anomaly_label' in df.columns:
            target_anomalies = df.loc[df.anomaly_label == 1]

            # Compute precision, recall, and F1 score
            true_positives = len(anomalies.merge(target_anomalies))
            false_positives = len(anomalies) - true_positives
            false_negatives = len(target_anomalies) - true_positives
            precision = true_positives / (true_positives + false_positives + 0.00001)
            recall = true_positives / (true_positives + false_negatives+ 0.00001)
            f1_score = 2 * precision * recall / (precision + recall+ 0.00001)

            with open(osp.join(out_dir, 'metric.json'), 'w') as f:
                json.dump({
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score
                }, f)

        # Impute anomalies
        imputed_df = impute_anomalies(
            df, 
            value_column, 
            anomalies
        )
        os.makedirs(osp.join(OUT_DIR, 'imputed'), exist_ok=True)
        imputed_df.to_csv(osp.join(OUT_DIR, 'imputed', filename), index=False)