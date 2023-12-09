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

def c_preprocess(all_df):
    ## Some series are missing timestamps
    ## We will fill in the missing timestamps with the mean value of left and right timestamps

    longest_index = np.argmax([len(df) for df in all_df])
    for i in range(len(all_df)):
        all_df[i]['date'] = pd.to_datetime(all_df[i]['date'])
        all_df[i] = all_df[i].set_index('date')

    longest_index = all_df[longest_index].index
    for i in range(len(all_df)):
        if len(all_df[i]) < len(longest_index):
            all_df[i] = all_df[i].reindex(longest_index)
            all_df[i].interpolate(limit_direction="both", inplace=True)

        all_df[i] = all_df[i].drop([
            'realtime_start', 'realtime_end'
        ], axis=1)
        all_df[i] = all_df[i].sort_index()
        all_df[i] = all_df[i].reset_index()
        # Count number of missing values
        countna = all_df[i].isna().sum().sum()
        if countna > 0:
            import pdb; pdb.set_trace()


    return all_df



def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    filenames = os.listdir(args.data_dir)

    if args.task == 'C':
        all_df = []

    for filename in filenames:
        filepath = osp.join(args.data_dir, filename)
        df = pd.read_csv(filepath)

        if args.task == 'ADP':
            df = adp_preprocess(df)
            df.to_csv(
                osp.join(args.out_dir, filename), 
                index=False
            )
        elif args.task == 'C':
            all_df.append(df)

    if args.task == 'C':
        all_df = c_preprocess(all_df)
        for filename, df in zip(filenames, all_df):
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
        new_df[time_column].max(), 
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
