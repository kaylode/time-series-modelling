import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import os.path as osp
import datetime
import yaml
from source.constants import TASK_COLUMNS
import argparse
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--out_dir', type=str, default='data/results')
parser.add_argument('--config_file', type=str, default='data/AD&P.yaml')
parser.add_argument('--task', type=str, choices=['AD&P', 'C'], default='AD&P')

def visualize_stl(original, trend, seasonal, resid, out_dir=None, figsize=(16,8)):

    plt.figure(figsize=figsize)
    plt.subplot(4,1,1)
    plt.plot(original)
    plt.title('Original Series', fontsize=16)
    plt.xticks(rotation=90)

    plt.subplot(4,1,2)
    plt.plot(trend)
    plt.title('Trend', fontsize=16)
    plt.xticks(rotation=90)

    plt.subplot(4,1,3)
    plt.plot(seasonal)
    plt.title('Seasonal', fontsize=16)
    plt.xticks(rotation=90)

    plt.subplot(4,1,4)
    plt.plot(resid)
    plt.title('Residual', fontsize=16)
    plt.xticks(rotation=90)

    plt.tight_layout()


    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(osp.join(out_dir, 'stl.png'), bbox_inches='tight')


def visualize_ts(
        df, time_column, value_column, 
        predictions=None, 
        targets = None,
        lower_bound=None, upper_bound=None,
        anomalies = None,
        outpath = None,
        freq='D',
        figsize=(16,12),
        zoom=4
    ):
    plt.figure(figsize=figsize)
    plt.subplots_adjust(left=0.05, right=0.95)  # Adjust left and right margins
    df[time_column] = pd.to_datetime(df[time_column])
    df = df.sort_values(by=time_column)
    plt.plot(df[time_column], df[value_column])

    if freq == 'D':
        time_dt = datetime.timedelta(days=1)
    elif freq == 'H':
        time_dt = datetime.timedelta(hours=1)
    elif freq == 'm':
        time_dt = datetime.timedelta(days=30)
    elif freq == 'Y':
        time_dt = datetime.timedelta(days=365)
    else:
        raise ValueError('Invalid frequency')
    
    date_range = pd.date_range(df[time_column].min(), df[time_column].max()+time_dt, freq=freq)
    plt.xticks(date_range)
    plt.xlabel('Timestamp')
    plt.xticks(rotation=90)
    plt.ylabel('KPI')

    for dt in date_range:
        plt.axvline(dt, color='k', linestyle='--', alpha=0.5)
    
    timestamp_diff_dt = df[time_column].iloc[1] - df[time_column].iloc[0]
    plt.title(f'Time Difference: {timestamp_diff_dt}')
    if predictions is not None:
        forecast_steps = len(predictions)
        forecast_timestamps = [
            df[time_column].max()+timestamp_diff_dt*i for i in range(1, forecast_steps+1)
        ]

        plt.plot(forecast_timestamps, predictions, color='g')

        if lower_bound is not None and upper_bound is not None:
            plt.fill_between(forecast_timestamps, lower_bound, upper_bound, color='g', alpha=0.1)

        # Zoom into the forecasted region
        plt.xlim([
            df[time_column].max()-timestamp_diff_dt*(forecast_steps*zoom), 
            df[time_column].max()+timestamp_diff_dt*(forecast_steps*2)
        ])

        plt.ylim([
            min(df[value_column].min(), predictions.min())-1, 
            max(df[value_column].max(), predictions.max())+1
        ])

    if targets is not None:
        plt.plot(targets[time_column], targets[value_column], color='r')


    if anomalies is not None:
        plt.scatter(
            anomalies[time_column], 
            anomalies[value_column], 
            color='r', marker='D'
        )

    if outpath is not None:
        dirname = os.path.dirname(outpath)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        plt.savefig(outpath)
        
    plt.close()

def visualize_autocorrelations(df, time_column, value_column, lags=None, out_dir=None):
    tmp_df = df.copy()
    tmp_df[time_column] = pd.to_datetime(tmp_df[time_column])
    tmp_df = tmp_df.set_index(time_column)

    # Calculate the autocorrelation function
    fig = plot_acf(tmp_df[value_column], lags=lags)
    plt.title(f'Lags: {lags}')
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(osp.join(out_dir, 'acf.png'), bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    # Calculate the partial autocorrelation function
    fig = plot_pacf(tmp_df[value_column], lags=lags)
    plt.title(f'Lags: {lags}')
    if out_dir is not None:
        plt.savefig(osp.join(out_dir, 'pacf.png'), bbox_inches='tight')
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    args = parser.parse_args()
    
    time_column = TASK_COLUMNS[args.task]['time_column']
    value_column = TASK_COLUMNS[args.task]['value_column']

    configs = yaml.load(open(args.config_file, 'r'), Loader=yaml.FullLoader)

    filenames = os.listdir(args.data_dir)
    for filename in tqdm(filenames):
        filepath = osp.join(args.data_dir, filename)
        df = pd.read_csv(filepath)
        df[time_column] = pd.to_datetime(df[time_column])
        filename = filename.split('.')[0]
        config = configs[filename]

        if 'anomaly_label' in df.columns:
            anomalies = df.loc[df.anomaly_label == 1]
        else:
            anomalies = None

        visualize_ts(
            df,
            freq = config['visualize_freq'],
            anomalies = anomalies,
            time_column=time_column, 
            value_column=value_column,
            outpath=osp.join(args.out_dir, 'original', f'{filename}.png')
        )

        visualize_autocorrelations(
            df,
            time_column=time_column, 
            value_column=value_column,
            lags=config.get('seasonality', None),
            out_dir=osp.join(args.out_dir, 'autocorrelations', filename)
        )

