import pandas as pd
import os
import os.path as osp
import yaml
import argparse
import matplotlib.pyplot as plt
import math

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from kneed import KneeLocator
from tsfresh import extract_features
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax
from source.constants import TASK_COLUMNS

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--out_dir', type=str, default='data/results')
parser.add_argument('--config_file', type=str, default='data/C.yaml')


def merge_files(data_dir):
    final_df = []
    filenames = os.listdir(data_dir)
    for filename in filenames:
        tsid = osp.splitext(filename)[0]
        filepath = osp.join(data_dir, filename)
        df = pd.read_csv(filepath)
        df['id'] = str(tsid)
        final_df.append(df)
    result_df = pd.concat(final_df, axis=0)
    return result_df

def feature_engineering(df, time_column, value_column, out_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    # Extract features
    features = extract_features(
        df, 
        column_id='id', 
        column_sort=time_column, 
        column_value=value_column
    )
    features = features.dropna(axis=1, how='any')
    features.to_csv(osp.join(out_dir, 'features.csv'))
    return features


def visualize_clusters(df, time_column, value_column, out_dir=None):
    pca = PCA(2)
    #Transform the data
    projected = pca.fit_transform(features)

    #Getting the Centroids
    centroids = kmeans.cluster_centers_

    for i, pred in enumerate(predictions):
        plt.scatter(projected[i, 0], projected[i, 1], label=pred)
    plt.legend()
    plt.show()

def fit_kmeans(n_clusters, features, model_configs={}, method:int = 0):
    if method == 0:
        kmeans = KMeans(n_clusters=n_clusters, **model_configs)
        kmeans.fit(features)
    elif method == 1:
        X_normalized = TimeSeriesScalerMinMax().fit_transform(features)
        kmeans = TimeSeriesKMeans(n_clusters=n_clusters, **model_configs)
        kmeans.fit(X_normalized)
    else:
        raise ValueError('Invalid method')
    return kmeans

def search_kmeans(features, model_configs, out_dir=None, method:int = 0):
    num_series = features.shape[0]

    # A list holds the SSE values and silhouette score for each k
    sse = []
    silhouette_coefficients = []

    for k in range(2, math.ceil(math.sqrt(num_series))+1):
        kmeans = fit_kmeans(k, features, model_configs, method=method)
        sse.append(kmeans.inertia_)
        score = silhouette_score(features, kmeans.labels_)
        silhouette_coefficients.append(score)

    optimal_k = visualize_elbow(sse, silhouette_coefficients, out_dir)
    return optimal_k

def visualize_elbow(sse, silhouette_coefficients, out_dir=None):
    num_series = len(sse) + 2
    kl = KneeLocator(range(2, num_series), sse, curve="convex", direction="decreasing")
    optimal_k = kl.elbow
    import pdb; pdb.set_trace()
    
    plt.figure(figsize=(10, 5))
    plt.style.use("fivethirtyeight")
    plt.plot(range(2, num_series), sse)
    plt.xticks(range(2, num_series))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.vlines(optimal_k, plt.ylim()[0], plt.ylim()[1], linestyles="--")
    plt.title(f"Optimal k: {optimal_k}")
    if out_dir:
        plt.savefig(osp.join(out_dir, 'elbow.png'))

    plt.figure(figsize=(10, 5))
    plt.style.use("fivethirtyeight")
    plt.plot(range(2, num_series), silhouette_coefficients)
    plt.xticks(range(2, num_series))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    if out_dir:
        plt.savefig(osp.join(out_dir, 'silhouette.png'))

    return optimal_k

def cluster(df, time_column, value_column, model_configs={}, out_dir=None):
    features = feature_engineering(
        df, time_column, value_column, out_dir=out_dir
    )

    print("Features shape: ", features.shape)

    optimal_k = search_kmeans(
        features, 
        model_configs=model_configs, 
        out_dir=out_dir,
        method=1
    )

    kmeans = fit_kmeans(optimal_k, features, model_configs, method=1)
    predictions = kmeans.predict(features)
    print(predictions)
    # visualize_clusters(
    #     features, 
    #     time_column, 
    #     value_column, 
    #     out_dir=out_dir
    # )


if __name__ == '__main__':

    args = parser.parse_args()
    DATA_DIR = args.data_dir
    OUT_DIR = args.out_dir
    task = 'C'
    time_column = TASK_COLUMNS[task]['time_column']
    value_column = TASK_COLUMNS[task]['value_column']

    # merge files
    merged_df = merge_files(args.data_dir)
    merged_df = merged_df.sort_values(by=['id', time_column])

    # load yaml file
    configs = yaml.load(open(args.config_file, 'r'), Loader=yaml.FullLoader)
    
    cluster(
        merged_df, time_column, value_column,
        model_configs=configs['model_configs'], 
        out_dir=osp.join(OUT_DIR, 'cluster')
    )