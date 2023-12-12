import pandas as pd
import os
import os.path as osp
import matplotlib.pyplot as plt
from tqdm import tqdm
from statsmodels.tsa.seasonal import STL
import json
import numpy as np
from source.visualize import visualize_ts, visualize_stl
from source.constants import TASK_COLUMNS
import argparse
import yaml

plt.style.use('fivethirtyeight')

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--out_dir', type=str, default='data/anomalies')
parser.add_argument('--config_file', type=str, default='data/AD&P.yaml')
parser.add_argument('--id', type=int, default=-1, required=False)

def detect_anomalies_rolling_mean(
        _df, time_column, value_column,
        window_size=10, threshold=2.0, 
        out_dir=None, visualize_freq='D',
        figsize=(16,4)
    ):

    # Compute rolling mean
    df = _df.copy()
    df['rolling_mean'] = df[value_column].rolling(window=window_size).mean()
    diff_rolling_mean = df['rolling_mean'].diff()

    diff_rolling_mean_mean = diff_rolling_mean.mean()
    diff_rolling_mean_std = diff_rolling_mean.std()
    lower = diff_rolling_mean_mean - threshold*diff_rolling_mean_std
    upper = diff_rolling_mean_mean + threshold*diff_rolling_mean_std

    # Plot rolling mean
    plt.figure(figsize=figsize)
    plt.plot(df[time_column], df['rolling_mean'], label='Rolling mean')
    plt.plot(df[time_column], diff_rolling_mean, label='Rolling mean residuals')
    plt.xticks(rotation=90)
    plt.fill_between(
        [df[time_column].min(), df[time_column].max()], 
        lower, upper, color='C3', alpha=0.25, 
        linestyle='--', linewidth=2
    )
    plt.legend()
    if out_dir is not None:
        plt.savefig(osp.join(out_dir, 'rolling_mean.png'), bbox_inches='tight')
    else:
        plt.show()

    # Mark anomalies based on threshold (you can adjust this threshold as needed)
    anomalies = df[(diff_rolling_mean < lower) | (diff_rolling_mean > upper)]

    visualize_ts(
        df, time_column, value_column, 
        outpath=osp.join(out_dir, 'anomalies_rolling_mean.png'),
        freq=visualize_freq,
        anomalies=anomalies,
        figsize=figsize
    )
    return anomalies

def detect_anomalies_rolling_std(
        _df, time_column, value_column,
        window_size, threshold=2.0, 
        out_dir=None, visualize_freq='D',
        figsize=(16,4)
    ):
    df = _df.copy()
    # Compute rolling standard deviation
    df['rolling_std'] = df[value_column].rolling(window=window_size).std()
    diff_rolling_std = df['rolling_std'].diff()

    diff_rolling_std_mean = diff_rolling_std.mean()
    diff_rolling_std_std = diff_rolling_std.std()
    lower = diff_rolling_std_mean - threshold*diff_rolling_std_std
    upper = diff_rolling_std_mean + threshold*diff_rolling_std_std

    # Plot rolling std
    plt.figure(figsize=figsize)
    plt.plot(df[time_column], df['rolling_std'], label='Rolling std')
    plt.plot(df[time_column], diff_rolling_std, label='Rolling std residuals')
    plt.xticks(rotation=90)
    plt.fill_between(
        [df[time_column].min(), df[time_column].max()], 
        lower, upper, color='C3', alpha=0.25, 
        linestyle='--', linewidth=2
    )
    plt.legend()
    if out_dir is not None:
        plt.savefig(osp.join(out_dir, 'rolling_std.png'), bbox_inches='tight')
    else:
        plt.show()

    # Mark anomalies based on threshold (you can adjust this threshold as needed)
    anomalies = df[(diff_rolling_std < lower) | (diff_rolling_std > upper)]
    visualize_ts(
        df, time_column, value_column, 
        outpath=osp.join(out_dir, 'anomalies_rolling_std.png'),
        freq=visualize_freq,
        anomalies=anomalies,
        figsize=figsize
    )
    return anomalies

def detect_anomalies_stl(
        df, time_column, value_column, 
        period=None,
        visualize_freq='D',
        threshold=3,
        out_dir:str=None,
        figsize=(16,16)
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
    plt.figure(figsize=(figsize[0], figsize[1]//4))
    plt.plot(df[value_column])
    plt.plot(estimated)
    if out_dir is not None:
        plt.savefig(osp.join(out_dir, 'estimated.png'), bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    # Detect anaomalies based on residuals
    resid_mu = resid.mean()
    resid_dev = resid.std()
    lower = resid_mu - threshold*resid_dev
    upper = resid_mu + threshold*resid_dev
    plt.figure(figsize=(figsize[0], figsize[1]//4))
    plt.plot(resid)
    plt.fill_between(
        [resid.index.min(), resid.index.max()], 
        lower, upper, color='C3', alpha=0.25, 
        linestyle='--', linewidth=2
    )
    if out_dir is not None:
        plt.savefig(osp.join(out_dir, 'residual.png'), bbox_inches='tight')
    else:
        plt.show()
    # Visualize anomalies
    anomalies = df[(resid < lower) | (resid > upper)]
    visualize_ts(
        df, time_column, value_column, 
        outpath=osp.join(out_dir, 'anomalies_stl.png'),
        freq=visualize_freq,
        anomalies=anomalies,
        figsize=(figsize[0], figsize[1]//4)
    )

    return anomalies

def impute_anomalies(df, value_column, anomalies, out_dir=None, figsize=(16,4)):
    # Impute new values for anomalies, 
    # using mean of window before and after anomaly
    new_df = df.copy()

    # Remove anomalies
    anomalies_idx = anomalies.index
    for idx in anomalies_idx:
        new_df.at[idx, value_column] = np.nan
    # Impute anomalies new values using interpolation
    new_df[value_column] = new_df[value_column].interpolate(limit_direction="both")
    
    # Plot only the difference, highlighting the difference parts
    plt.figure(figsize=figsize)
    plt.plot(new_df[value_column], label='Imputed', color='C1')
    plt.plot(df[value_column], label='Original', color='C0', alpha=0.5)
    plt.legend()

    if out_dir is not None:
        plt.savefig(osp.join(out_dir, 'imputed.png'), bbox_inches='tight')
    else:
        plt.show()

    # Clear memory of matplotlib
    plt.cla()
    plt.clf()
    plt.close('all')
    
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

    if args.id >= 0:
        filenames = [f'dataset_{args.id}.csv']
    else:
        filenames = sorted(os.listdir(DATA_DIR), key=lambda x: int(x.split('.')[0].split('_')[-1]))
    
    for filename in (pbar := tqdm(filenames)):
        pbar.set_description(f"Processing {filename}")
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

        # Clear memory of matplotlib
        plt.cla()
        plt.clf()
        plt.close('all')

        # Resample to frequency
        df[time_column] = pd.to_datetime(df[time_column])
        df = df.sort_values(by=time_column)
        df = df.set_index(time_column)
        anomalies = anomalies.set_index(time_column)
        df = df.resample(config['freq']).mean()

        # Impute anomalies
        imputed_df = impute_anomalies(
            df, 
            value_column, 
            anomalies,
            out_dir=out_dir
        )
        imputed_df = imputed_df.reset_index()
        os.makedirs(osp.join(OUT_DIR, 'imputed'), exist_ok=True)
        anomalies.to_csv(osp.join(OUT_DIR, file_prefix, 'anomalies.csv'))
        imputed_df.to_csv(osp.join(OUT_DIR, 'imputed', filename), index=False)