import numpy as np
import pandas as pd
import os
import os.path as osp
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf

from source.visualize import visualize_ts, visualize_stl
from source.constants import TASK_COLUMNS
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--out_dir', type=str, default='data/anomalies')
parser.add_argument('--config_file', type=str, default='data/AD&P.yaml')



def plot_acf_(df, outname=None):
    # Assuming df is your DataFrame with 'timestamp' and 'kpi_value' columns
    # Make sure 'timestamp' is in datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Set 'timestamp' as the index
    df = df.set_index('timestamp')

    # Calculate the autocorrelation function
    fig = plot_acf(df['kpi_value'])

    if outname is not None:
        plt.savefig(f'{outname}.png', bbox_inches='tight')
    else:
        plt.show()

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
   
if __name__ == '__main__':

    args = parser.parse_args()
    DATA_DIR = args.data_dir
    OUT_DIR = args.out_dir
    task = 'AD&P'
    time_column = TASK_COLUMNS[task]['time_column']
    value_column = TASK_COLUMNS[task]['value_column']

    # load yaml file
    configs = yaml.load(open(args.config_file, 'r'), Loader=yaml.FullLoader)

    filenames = ['dataset_1.csv', 'dataset_2.csv', 'dataset_3.csv'] #os.listdir(DATA_DIR)
    for filename in tqdm(filenames):
        filepath = osp.join(DATA_DIR, filename)
        file_prefix = filename.split('.')[0]
        config = configs[file_prefix]
        df = pd.read_csv(filepath)
        df[time_column] = pd.to_datetime(df[time_column])

        out_dir = osp.join(OUT_DIR, file_prefix)
        os.makedirs(out_dir, exist_ok=True)
        if config['anomalies']['method'] == 'stl':
            detect_anomalies_stl(
                df, time_column, value_column,
                period=config['seasonality'],
                visualize_freq=config['visualize_freq'],
                threshold=config['anomalies']['threshold'],
                out_dir=out_dir
            )
        elif config['anomalies']['method'] == 'rolling_mean':
            detect_anomalies_rolling_mean(
                df, time_column, value_column,
                window_size=config['anomalies']['window_size'], 
                threshold=config['anomalies']['threshold'], 
                visualize_freq=config['visualize_freq'],
                out_dir=out_dir
            )
        elif config['anomalies']['method'] == 'rolling_std':
            detect_anomalies_rolling_std(
                df, time_column, value_column,
                window_size=config['anomalies']['window_size'], 
                threshold=config['anomalies']['threshold'], 
                visualize_freq=config['visualize_freq'],
                out_dir=out_dir    
            )
        else:
            raise NotImplementedError