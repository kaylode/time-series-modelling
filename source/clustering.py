import pandas as pd
import os
import os.path as osp
import yaml
import argparse
import matplotlib.pyplot as plt
import numpy as np
import joblib

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from tsfresh import extract_features
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax
from source.constants import TASK_COLUMNS
from source.visualize import visualize_pca, visualize_series_cluster, visualize_elbow

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

def feature_engineering(df, time_column, value_column, out_dir=None, method:str='norm'):
    os.makedirs(out_dir, exist_ok=True)

    features = None
    id_names = []

    # Normalize first, using min, max scale
    unique_ids = df.id.unique()
    for i in range(len(unique_ids)):
        series = df[df.id == unique_ids[i]][value_column].values
        series = TimeSeriesScalerMinMax().fit_transform(series.reshape(1, -1))
        df.loc[df.id == unique_ids[i], value_column] = series.reshape(-1)

    # Extract features using tsfresh, convert from time series to feature matrix
    if method == 'feature_matrix':
        features = extract_features(
            df, 
            column_id='id', 
            column_sort=time_column, 
            column_value=value_column
        )
        features = features.dropna(axis=1, how='any')
        features.to_csv(osp.join(out_dir, 'features.csv'))
        id_names = features.index.values

    # Convert from time series to feature matrix
    elif method == 'norm':
        features = []
        for i in range(len(unique_ids)):
            series = df[df.id == unique_ids[i]][value_column].values
            features.append(series.reshape(-1))
            id_names.append(unique_ids[i])
    else:
        raise ValueError('Invalid method')
    return id_names, features

def fit_kmeans(n_clusters, features, model_configs={}, method:str = 'kmeans'):
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, **model_configs)
        kmeans.fit(features)
    elif method == 'tskmeans':
        kmeans = TimeSeriesKMeans(n_clusters=n_clusters, **model_configs)
        kmeans.fit(features)
    else:
        raise ValueError('Invalid method')
    return kmeans

def search_kmeans(features, model_configs, out_dir=None, method:str = 'kmeans'):
    if isinstance(features, list):
        num_series = len(features)
    else:
        num_series = features.shape[0]

    # A list holds the SSE values and silhouette score for each k
    sse = []
    silhouette_coefficients = []

    for k in range(2, num_series):
        kmeans = fit_kmeans(k, features, model_configs, method=method)
        sse.append(kmeans.inertia_)
        score = silhouette_score(features, kmeans.labels_)
        silhouette_coefficients.append(score)

    optimal_k = visualize_elbow(sse, silhouette_coefficients, out_dir)
    return optimal_k

def cluster(df, time_column, value_column, fe_method:str = 'norm', method:str = 'kmeans', model_configs={}, out_dir=None):
    # Feature engineering
    id_names, features = feature_engineering(
        df, time_column, value_column, out_dir=out_dir,
        method=fe_method
    )

    if method in ['kmeans', 'tskmeans']:
        # Search for optimal k
        optimal_k = search_kmeans(
            features, 
            model_configs=model_configs, 
            out_dir=out_dir,
            method=method
        )
        model = fit_kmeans(optimal_k, features, model_configs, method=method)
        centroids = model.cluster_centers_.squeeze(-1)
    
        # Save model
        joblib.dump(model, osp.join(out_dir, 'model.pkl'))
    else:
        raise ValueError('Invalid method')

    # Make predictions
    predictions = model.predict(features)
    prediction_mapping = {
        k:v for k,v in zip(id_names, predictions)
    }
    tmp_df = df.copy()
    tmp_df['cluster'] = tmp_df['id'].map(prediction_mapping)
    
    # Visualize predictions
    visualize_pca(
        np.array(features), 
        predictions, 
        out_dir=out_dir,
        centroids=centroids
    )

    # Visualize series clusters
    tmp_df = tmp_df.set_index(time_column)
    visualize_series_cluster(
        tmp_df, 
        out_dir=out_dir
    )


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
        fe_method=configs['fe_method'],
        method=configs['cluster_method'],
        model_configs=configs['model_configs'], 
        out_dir=osp.join(OUT_DIR, 'cluster')
    )