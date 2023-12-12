import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm
import os
import os.path as osp
import datetime
import yaml
import math
from source.constants import TASK_COLUMNS
import argparse
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.decomposition import PCA
from kneed import KneeLocator

plt.style.use('fivethirtyeight')

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--out_dir', type=str, default='data/results')
parser.add_argument('--config_file', type=str, default='data/AD&P.yaml')
parser.add_argument('--task', type=str, choices=['AD&P', 'C'], default='AD&P')

def visualize_diff(df, time_column, value_column, periods, out_dir=None, figsize=(16,12)):
    tmp_df = df.copy()
    tmp_df[time_column] = pd.to_datetime(tmp_df[time_column])
    tmp_df = tmp_df.set_index(time_column)
    tmp_df = tmp_df[value_column]

    # Visualize 1st differencing
    diff = tmp_df.diff().dropna()
    plt.figure(figsize=figsize)
    plt.subplot(3,1,1)
    plt.plot(diff, color='C0', label='1st Differencing')
    plt.title('1st Differencing')

    # Visualize 2st differencing
    diff2 = diff.diff().dropna()
    plt.subplot(3,1,2)
    plt.plot(diff2, color='C0', label='2nd Differencing')
    plt.title('2nd Differencing')

    # Visualize seasonal differencing
    seasonal_diff = tmp_df.diff(periods=periods).dropna()
    plt.subplot(3,1,3)
    plt.plot(seasonal_diff, color='C0', label='Seasonal Differencing')
    plt.title('Seasonal Differencing')

    plt.legend(loc='best')

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(osp.join(out_dir, 'diff.png'), bbox_inches='tight')
    else:
        plt.show()
    plt.clf()
    plt.close()


def visualize_rollings(
        df, time_column, value_column,
        window_size, out_dir=None, figsize=(16,8)
    ):
    tmp_df = df.copy()
    tmp_df[time_column] = pd.to_datetime(tmp_df[time_column])
    tmp_df = tmp_df.set_index(time_column)
    tmp_df = tmp_df[value_column]

    # Visualize rolling mean and std
    rolling_mean = tmp_df.rolling(window=window_size).mean()
    rolling_std = tmp_df.rolling(window=window_size).std()
    plt.figure(figsize=figsize)
    plt.plot(tmp_df, label='Original')
    plt.plot(rolling_mean, color='C1', label='Rolling Mean')
    plt.plot(rolling_std, color='C3', label='Rolling Std')
    plt.legend(loc='best')

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(osp.join(out_dir, 'rolling_stats.png'), bbox_inches='tight')
    else:
        plt.show()
    plt.clf()
    plt.close()

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
    else:
        plt.show()

    plt.close()

def check_stationary(df, value_column):
    # Check if time series is stationary using Augmented Dickey-Fuller test
    try:
        result = adfuller(df[value_column].dropna())
    except ValueError:
        result = (0,0)
    return result # lower mean stastically significant, reject null hypothesis


def visualize_ts(
        df, time_column, value_column, 
        predictions=None, 
        targets = None,
        lower_bound=None, upper_bound=None,
        anomalies = None,
        outpath = None,
        freq='D',
        figsize=(16,12),
        check_stationarity=False,
        plot_legend_labels={
            'df': 'Original', 
            'predictions': 'Predictions', 
            'targets': 'Targets', 
            'anomalies': 'Anomalies'
        }
    ):
    plt.figure(figsize=figsize)
    plt.subplots_adjust(left=0.05, right=0.95)  # Adjust left and right margins
    df[time_column] = pd.to_datetime(df[time_column])
    df = df.sort_values(by=time_column)

    if targets is not None:
        plt.plot(targets[time_column], targets[value_column], color='C2', label=plot_legend_labels.get('targets', None))

    plt.plot(df[time_column], df[value_column], color='C0', label=plot_legend_labels.get('df', None))

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
    plt.ylabel(value_column)

    for dt in date_range:
        plt.axvline(dt, color='k', linestyle='--', alpha=0.5)
    
    title = ""
    if check_stationarity:
        sta = check_stationary(df, value_column)
        title += f'p-value: {sta[1]}\n'

    title += f'Frequency: {freq}\n'
    
    if predictions is not None:
        plt.plot(predictions, color='C3', label=plot_legend_labels.get('predictions', None), alpha=0.7)

        if lower_bound is not None and upper_bound is not None:
            plt.fill_between(predictions.index, lower_bound, upper_bound, color='g', alpha=0.1)


    if anomalies is not None:
        plt.scatter(
            anomalies[time_column], 
            anomalies[value_column], 
            color='C1', marker='D',
            label=plot_legend_labels.get('anomalies', None)
        )

    plt.title(title)
    plt.legend(loc='best')
    if outpath is not None:
        dirname = os.path.dirname(outpath)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        plt.savefig(outpath)
    else:
        plt.show()
    
    plt.clf()
    plt.close()

def visualize_autocorrelations(df, time_column, value_column, lags=None, out_dir=None):
    tmp_df = df.copy()
    tmp_df[time_column] = pd.to_datetime(tmp_df[time_column])
    tmp_df = tmp_df.set_index(time_column)

    # Calculate the autocorrelation function and partial autocorrelation function
    fig, ax = plt.subplots(2,1, figsize=(16,8))

    plot_acf(tmp_df[value_column], lags=lags, ax=ax[0])
    plot_pacf(tmp_df[value_column], lags=lags, ax=ax[1])
    ax[0].set_title('Autocorrelation')
    ax[1].set_title('Partial Autocorrelation')
    plt.title(f'Lags: {lags}')
    plt.tight_layout()

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(osp.join(out_dir, 'autocorrelation.png'), bbox_inches='tight')
    else:
        plt.show()
    plt.clf()
    plt.close()

"""
VISUALIZATION FUNCTIONS FOR CLUSTERING TASK
"""

def visualize_series_cluster(series_df, out_dir=None, figsize=(25,25)):
    labels = series_df.cluster.unique()
    num_clusters = len(set(labels))
    plot_count = math.ceil(math.sqrt(num_clusters))

    fig = plt.figure(figsize=figsize)
    fig.suptitle('Clusters')

    spec = gridspec.GridSpec(plot_count, plot_count, figure=fig)

    for i, label in enumerate(set(labels)):
        series_ids = series_df.loc[series_df.cluster == label].id.unique()
        row_i, col_j = divmod(i, plot_count)
        axs = fig.add_subplot(spec[row_i, col_j])

        cluster = []
        for sid in series_ids:
            tmp_df = series_df.loc[series_df.id==sid]
            axs.plot(tmp_df.value, c="gray", alpha=0.4)
            cluster.append(tmp_df.value)
        if len(cluster) > 0:
            # caculate average
            avg_df = pd.concat(cluster, axis=1).mean(axis=1)
            axs.plot(avg_df, c="red")
        axs.set_title("Cluster " + str(label))
    # Adjust layout to prevent clipping of titles
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(osp.join(out_dir, 'clusters.png'), bbox_inches='tight')
    else:
        plt.show()
    plt.cla()
    plt.clf()
    plt.close()

def visualize_pca(features, labels, centroids=None, out_dir=None):
    pca = PCA(2)
    projected = pca.fit_transform(features)
    unique_labels = list(set(labels))
    for label in unique_labels:
        projected_label = projected[labels == label]
        plt.scatter(projected_label[:, 0], projected_label[:, 1], label=label)
    if centroids is not None:
        centroids = pca.transform(centroids)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.1, label='Centroids')
    
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(osp.join(out_dir, 'pca.png'), bbox_inches='tight')
    else:
        plt.show()
    plt.cla()
    plt.clf()
    plt.close()

def visualize_elbow(sse, silhouette_coefficients, out_dir=None):
    num_series = len(sse) + 2
    kl = KneeLocator(range(2, num_series), sse, curve="convex", direction="decreasing")
    optimal_k = kl.elbow
    
    plt.figure(figsize=(10, 5))
    plt.style.use("fivethirtyeight")
    plt.plot(range(2, num_series), sse)
    plt.xticks(range(2, num_series))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")

    if optimal_k:
        plt.vlines(optimal_k, plt.ylim()[0], plt.ylim()[1], linestyles="--")
        plt.title(f"Optimal k: {optimal_k}")
    if out_dir:
        plt.savefig(osp.join(out_dir, 'elbow.png'))
    else:
        plt.show()

    plt.figure(figsize=(10, 7))
    plt.style.use("fivethirtyeight")
    plt.plot(range(2, num_series), silhouette_coefficients)
    plt.xticks(range(2, num_series))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    if out_dir:
        plt.savefig(osp.join(out_dir, 'silhouette.png'))
    else:
        plt.show()

    plt.cla()
    plt.clf()
    plt.close()

    return optimal_k


def visualize_grid(
        df, figsize=(30, 20), 
        outpath=None, 
        time_column='date', value_column='value',
        predictions=None, 
        anomalies=None
    ):
    fig = plt.figure(figsize=figsize)

    df_ = df.copy()
    predictions_ = predictions.copy() if predictions is not None else None
    anomalies_ = anomalies.copy() if anomalies is not None else None

    ids = df_.id.unique()
    columns = int(math.sqrt(len(ids)))
    rows = math.ceil(len(ids) / columns)

    for i in range(0, columns*rows):
        if i >= len(ids):
            break
        df = df_.loc[df_.id==ids[i]]
        df = df.sort_values(by=time_column)
        df = df.set_index(time_column)

        if predictions_ is not None:
            pred_df = predictions_.loc[predictions_.id==ids[i]]
            pred_df = pred_df.sort_values(by=time_column)
            pred_df = pred_df.set_index(time_column)

        if anomalies_ is not None:
            anomalies = anomalies_.loc[anomalies_.id==ids[i]]
            anomalies = anomalies.sort_values(by=time_column)
            anomalies = anomalies.set_index(time_column)

        fig.add_subplot(rows, columns, i+1)

        # remove all the ticks (both axes), and tick labels on the Y axis
        plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
        

        if predictions_ is not None:
            plt.plot(df[value_column],  label='Ground Truth', alpha=0.7)
            plt.plot(pred_df['predicted_mean'], color='C3', label='Predictions', alpha=0.7)
            plt.fill_between(pred_df.index, pred_df['lower y'], pred_df['upper y'], color='C3', alpha=0.1)

        if anomalies_ is not None:
            # Plot
            plt.plot(df[value_column],  label='Original')
            plt.scatter(
                anomalies.index,
                anomalies[value_column], 
                color='C1', marker='D',
                label='Anomalies'
            )

        # add figure name below each image as title
        plt.title(ids[i], fontsize=8)

    if outpath:
        plt.savefig(outpath)

    else:
        plt.show()

    plt.clf()
    plt.close()

if __name__ == "__main__":
    args = parser.parse_args()
    
    time_column = TASK_COLUMNS[args.task]['time_column']
    value_column = TASK_COLUMNS[args.task]['value_column']

    configs = yaml.load(open(args.config_file, 'r'), Loader=yaml.FullLoader)

    filenames = sorted(os.listdir(args.data_dir))

    for filename in (pbar := tqdm(filenames)):
        pbar.set_description(f"Processing {filename}")
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
            check_stationarity=True,
            outpath=osp.join(args.out_dir, 'original', f'{filename}.png'),
        )

        visualize_autocorrelations(
            df,
            time_column=time_column, 
            value_column=value_column,
            lags=config.get('seasonality', 288),
            out_dir=osp.join(args.out_dir, 'stats', filename)
        )

        visualize_rollings(
            df,
            time_column=time_column,
            value_column=value_column,
            window_size=config.get('seasonality', 288),
            out_dir=osp.join(args.out_dir, 'stats', filename)
        )

        visualize_diff(
            df,
            time_column=time_column,
            value_column=value_column,
            periods=config.get('seasonality', 288),
            out_dir=osp.join(args.out_dir, 'stats', filename)
        )

        # Clear memory of matplotlib
        plt.cla()
        plt.clf()
        plt.close('all')