import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import os.path as osp
import datetime

def visualize_ts(
        df, time_column, value_column, 
        predictions=None, 
        lower_bound=None, upper_bound=None,
        outpath = None
    ):
    plt.figure(figsize=(16,12))
    plt.subplots_adjust(left=0.05, right=0.95)  # Adjust left and right margins
    df[time_column] = pd.to_datetime(df[time_column])
    df = df.sort_values(by=time_column)
    plt.plot(df[time_column], df[value_column])

    date_range = pd.date_range(df.timestamp.min(), df.timestamp.max()+datetime.timedelta(days=1), freq='D')
    plt.xticks(date_range)
    plt.xlabel('Timestamp')
    plt.xticks(rotation=90)
    plt.ylabel('KPI')

    for dt in date_range:
        plt.axvline(dt, color='k', linestyle='--', alpha=0.5)
    
    timestamp_diff_dt = df.timestamp.iloc[1] - df.timestamp.iloc[0]
    plt.title(f'Time Difference: {timestamp_diff_dt}')
    if predictions is not None:
        forecast_steps = len(predictions)
        forecast_timestamps = [
            df.timestamp.max()+timestamp_diff_dt*i for i in range(1, forecast_steps+1)
        ]

        plt.plot(forecast_timestamps, predictions, color='g')

        if lower_bound is not None and upper_bound is not None:
            plt.fill_between(forecast_timestamps, lower_bound, upper_bound, color='g', alpha=0.1)

    if outpath is not None:
        dirname = os.path.dirname(outpath)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        plt.savefig(outpath)
        
    plt.close()

if __name__ == "__main__":
    TASK = 'AD&P'
    DATA_DIR = f'/home/mpham/workspace/huawei-time-series/data/processed/{TASK}'

    filenames = os.listdir(DATA_DIR)
    for filename in filenames:
        filepath = osp.join(DATA_DIR, filename)
        df = pd.read_csv(filepath)
        filename = filename.split('.')[0]

        visualize_ts(
            df,
            time_column='timestamp', 
            value_column='kpi_value',
            outpath=f'/home/mpham/workspace/huawei-time-series/results/{TASK}/figures/{filename}_raw.png'
         )