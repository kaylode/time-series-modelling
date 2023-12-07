import pandas as pd
import os
import os.path as osp
import argparse
import numpy as np
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--out_dir', type=str, default='data/processed')
parser.add_argument('--task', type=str, choices=['AD&P', 'C'], default='ADP')

def adp_preprocess(df):
    result_df = df.sort_values(['timestamp'])
    return result_df

def c_preprocess(df):
    result_df = df.drop([
        'realtime_start', 'realtime_end'
    ], axis=1)
    result_df = result_df.sort_values(['date'])
    return result_df


def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    filenames = os.listdir(args.data_dir)
    for filename in filenames:
        filepath = osp.join(args.data_dir, filename)
        df = pd.read_csv(filepath)

        if args.task == 'ADP':
            df = adp_preprocess(df)
        elif args.task == 'C':
            df = c_preprocess(df)
    
        df.to_csv(
            osp.join(args.out_dir, filename), 
            index=False
        )

    return df


def normalize_ts(df, value_column, out_column):
    new_df = df.copy()
    mu = np.mean(new_df[value_column])
    sigma = np.std(new_df[value_column])
    new_df[out_column] = new_df.apply(lambda x: (x[value_column] - mu) / sigma, axis=1)
    return new_df, mu, sigma

def missing_timestamps(df, time_column, freq='D'):
    new_df = df.copy()
    new_df[time_column] = pd.to_datetime(new_df[time_column])
    new_df = new_df.sort_values(by=time_column)
    date_range = pd.date_range(
        new_df[time_column].min(), 
        new_df[time_column].max()+datetime.timedelta(days=1), 
        freq=freq
    )
    missing_timestamps = []
    for dt in date_range:
        if dt not in new_df[time_column].values:
            missing_timestamps.append(dt)
    return missing_timestamps



if __name__ == '__main__':
    args = parser.parse_args()
    df = run(args)
