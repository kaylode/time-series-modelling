import os
import os.path as osp
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import pandas as pd


plt.style.use('fivethirtyeight')

def visualize_grid(
        df_, figsize=(30, 20), 
        outpath=None, 
        time_column='date', value_column='value',
        predictions_=None, 
        anomalies_=None
    ):
    fig = plt.figure(figsize=figsize)
    columns = 9
    rows = 3

    ids = df_.id.unique()

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
            plt.plot(pred_df['predicted_mean'], color='orange', label='Predictions', alpha=0.7)
            plt.fill_between(pred_df.index, pred_df['lower y'], pred_df['upper y'], color='orange', alpha=0.1)


        if anomalies_ is not None:
            # Plot
            plt.plot(df[value_column],  label='Original')
            plt.scatter(
                anomalies.index,
                anomalies[value_column], 
                color='r', marker='D',
                label='Anomalies'
            )

        # add figure name below each image as title
        plt.title(ids[i], fontsize=8)

    if outpath:
        plt.savefig(outpath)




if __name__ == '__main__':

    DATA_DIR = 'data/InterviewCaseStudies/AD&P'
    PROCESSED_DATA_DIR = 'data/processed/AD&P'
    IMPUTED_DATA_DIR = 'results/AD&P/anomalies/imputed'
    ANOMALIES = 'results/AD&P/anomalies/{DATANAME}/anomalies.csv'
    FORECAST = 'results/AD&P/forecast/{DATANAME}/predictions.csv'
    CONFIG_FILE = 'configs/AD&P.yaml'
    OUT_DIR = 'results/AD&P/report'
    
    filenames = sorted(os.listdir(DATA_DIR), key=lambda x: int(x.split('.')[0].split('_')[-1]))
    configs = yaml.load(open(CONFIG_FILE, 'r'), Loader=yaml.FullLoader)
    os.makedirs(OUT_DIR, exist_ok=True)
    time_column = 'timestamp'
    value_column = 'kpi_value'

    dfs = []
    anomalies_dfs = []
    forecast_dfs = []
    
    for filename in (pbar := tqdm(filenames)):
        pbar.set_description(f"Processing {filename}")
        file_prefix = filename.split('.')[0]
        config = configs[file_prefix]

        filepath = osp.join(DATA_DIR, filename)
        imputed_filepath = osp.join(IMPUTED_DATA_DIR, filename)
        anomalies_path = ANOMALIES.format(DATANAME=file_prefix)
        forecast_path = FORECAST.format(DATANAME=file_prefix)

        df = pd.read_csv(filepath)
        anomalies = pd.read_csv(anomalies_path)
        forecast = pd.read_csv(forecast_path)

        df[time_column] = pd.to_datetime(df[time_column])
        anomalies[time_column] = pd.to_datetime(anomalies[time_column])
        forecast[time_column] = pd.to_datetime(forecast[time_column])

        df['id'] = str(file_prefix)
        anomalies['id'] = str(file_prefix)
        forecast['id'] = str(file_prefix)

        dfs.append(df)
        anomalies_dfs.append(anomalies)
        forecast_dfs.append(forecast)

    dfs = pd.concat(dfs,axis=0)
    anomalies_dfs = pd.concat(anomalies_dfs,axis=0)
    forecast_dfs = pd.concat(forecast_dfs,axis=0)


    visualize_grid(
        dfs, figsize=(45, 15), 
        outpath=osp.join(OUT_DIR, 'anomalies.png'),
        time_column=time_column, value_column=value_column,
        anomalies_=anomalies_dfs
    )

    visualize_grid(
        dfs, figsize=(45, 15), 
        outpath=osp.join(OUT_DIR, 'forecast.png'),
        time_column=time_column, value_column=value_column,
        predictions_=forecast_dfs, 
    )