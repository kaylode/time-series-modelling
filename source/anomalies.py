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

from datetime import datetime
from source.visualize import visualize_ts
from source.constants import TASK_COLUMNS
from source.preprocess import missing_timestamps
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


def detect_anomalies(df, config={}):

    train_df = df.copy()
    train_df = train_df.sort_values(by='timestamp')
    train_df = train_df.set_index('timestamp')
    train_df = train_df['kpi_value']
    seasonality = config.get('seasonality', None)
    freq = config.get('freq', None)

    # mts = missing_timestamps(df, 'timestamp', freq=freq)
    # plot_acf_(df, outname=None)

    import pdb; pdb.set_trace()

    stl = STL(train_df, seasonal=seasonality)
    result = stl.fit()
    seasonal, trend, resid = result.seasonal, result.trend, result.resid
    import pdb; pdb.set_trace()

    # plt.figure(figsize=(8,6))

    # plt.subplot(4,1,1)
    # plt.plot(df['value'])
    # plt.title('Original Series', fontsize=16)

    # plt.subplot(4,1,2)
    # plt.plot(trend)
    # plt.title('Trend', fontsize=16)

    # plt.subplot(4,1,3)
    # plt.plot(seasonal)
    # plt.title('Seasonal', fontsize=16)

    # plt.subplot(4,1,4)
    # plt.plot(resid)
    # plt.title('Residual', fontsize=16)

    # plt.tight_layout()



if __name__ == '__main__':

    args = parser.parse_args()
    DATA_DIR = args.data_dir
    OUT_DIR = args.out_dir
    task = 'AD&P'
    time_column = TASK_COLUMNS[task]['time_column']
    value_column = TASK_COLUMNS[task]['value_column']
    freq = TASK_COLUMNS[task]['freq']

    # load yaml file
    configs = yaml.load(open(args.config_file, 'r'), Loader=yaml.FullLoader)

    filenames = ['dataset_1.csv'] #os.listdir(DATA_DIR)
    for filename in tqdm(filenames):
        filepath = osp.join(DATA_DIR, filename)
        file_prefix = filename.split('.')[0]
        config = configs[file_prefix]
        df = pd.read_csv(filepath)
        df[time_column] = pd.to_datetime(df[time_column])
        detect_anomalies(df, config)
        break

