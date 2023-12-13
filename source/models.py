from typing import *
import os
import os.path as osp
import json
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
from prophet import Prophet
import optuna
from optuna.visualization import (
    plot_contour,
    plot_edf,
    plot_intermediate_values,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_slice,
)
from optuna.storages import JournalStorage, JournalFileStorage
from source.visualize import visualize_ts

def postprocess(
        pred_df, 
        last_indexes=None, last_values=None, 
        sigma=None, mu=None, 
        cumsum=None, cumsum_periods=1
    ):
    # Reverse differencing
    if cumsum: #assuming df is already cumsumed
        freq = pred_df.index.freq
        if last_indexes is not None and last_values is not None:
            for col in pred_df.columns:
                for last_index, last_value in zip(last_indexes, last_values):
                    if last_index in pred_df.index:
                        continue
                    pred_df.at[last_index, col] = last_value
        pred_df = pred_df.sort_index()
        pred_df = pred_df.asfreq(freq)
        # Customized cumsum
        # index is datetime
        for index in pred_df.index[cumsum_periods:]:
            for col in pred_df.columns:
                pred_df.at[index, col] = (
                    pred_df.at[index, col] 
                    + pred_df.at[index-pred_df.index.freq*cumsum_periods, 'predicted_mean']
                )
        if last_indexes is not None:
            pred_df = pred_df.drop(index=last_indexes)
        pred_df = pred_df.asfreq(freq)

    # Reverse normalization
    if sigma is not None and mu is not None:
        pred_df = pred_df * sigma + mu

    return pred_df

def feature_engineering(df, value_column, method='diff', mu=None, sigma=None, seasonal_lag=None):

    assert method in ['none', 'diff', 'diff2', 'seasonal_diff', 'norm'], f"Invalid method: {method}"

    def normalize_ts(df, value_column, mu=None, sigma=None):
        if mu is None:
            mu = np.mean(df[value_column])
        if sigma is None:
            sigma = np.std(df[value_column])

        if sigma == 0:
            sigma = 1e-6
        norm_values = df.apply(lambda x: (x[value_column] - mu) / sigma, axis=1)
        return norm_values, mu, sigma

    norm_values, mu, sigma = normalize_ts(df, value_column, mu, sigma)
    if method == 'diff':
        engineered_values = norm_values.diff()
    elif method == 'diff2':
        engineered_values = norm_values.diff()
        engineered_values = engineered_values.diff()
    elif method == 'seasonal_diff':
        engineered_values = norm_values.diff(periods=seasonal_lag)
    elif method == 'none':
        engineered_values = df[value_column]
        mu, sigma = None, None
    elif method == 'norm':
        engineered_values = norm_values
    else:
        raise NotImplementedError

    return norm_values, engineered_values, mu, sigma


def fit_arima(df, model_configs={}):
    order = (model_configs['p'], model_configs.get('d', 0), model_configs['q'])
    # Fit ARIMA model
    model = ARIMA(df.dropna(), order=order)
    fit_model = model.fit() 

    # print(fit_model.summary())
    return fit_model

def predict_arima(fit_model, num_predictions=5):
    fitted_values = fit_model.fittedvalues
    predictions = fit_model.get_prediction(0, fitted_values.shape[0] + num_predictions)
    yhat = predictions.predicted_mean
    yhat_conf_int = predictions.conf_int(alpha=0.05)
    pred_df = pd.merge(yhat.reset_index(), yhat_conf_int.reset_index(), on='index')
    pred_df = pred_df.set_index('index')
    pred_df = pred_df.asfreq(fitted_values.index.freq)
    return pred_df

def fit_prophet(df, model_configs={}):
    model = Prophet(**model_configs)
    model.fit(df)
    return model

def predict_prophet(model, num_predictions=5):
    future = model.make_future_dataframe(periods=num_predictions)
    predictions = model.predict(future)
    result = predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    result = result.set_index('ds')
    result = result.rename(columns={'yhat': 'predicted_mean', 'yhat_lower': 'lower y', 'yhat_upper': 'upper y'})
    
    return result.tail(num_predictions)


def validate(actual, forecast):
    mape = np.mean(np.abs(forecast - actual)/(np.abs(actual)+1))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/(actual+1))   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr

    return({
        'mape':mape, 'me':me, 'mae': mae, 
        'mpe': mpe, 'rmse':rmse,  
        'corr':corr, 
    })

def get_mean_std(scores):
    return (
        np.mean(scores), \
        np.std(scores)
    )

def objective(trial, df, params):
    seasonal_lag = params['seasonal_lag']
    method = params['method']
    fe_method = params.get('fe_method', None)
    num_predictions = params['num_predictions']
    n_splits = params['n_splits']
    best_key = params['best_key']
    time_column = params['time_column']
    value_column = params['value_column']
    
    if n_splits > 1:
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=num_predictions)
        indexes = tscv.split(df)
    else:
        train_index = list(range(df.shape[0] - num_predictions))
        test_index = list(range(df.shape[0] - num_predictions, df.shape[0]))
        indexes = [(train_index, test_index)]
    scores = []

    # Split the time series using TimeSeriesSplit
    for fold_id, (train_index, test_index) in enumerate(indexes):
        train_df, test_df = df.iloc[train_index], df.iloc[test_index]

        if fe_method is None:
            fe_method = trial.suggest_categorical("fe_method", ["diff", "seasonal_diff", "norm"])
        (
            norm_train, 
            engineered_train, 
            mu, 
            sigma
        ) = feature_engineering(
            train_df, 
            value_column,
            method=fe_method,
            seasonal_lag=seasonal_lag
        )

        if method == 'arima':

            if fe_method == 'norm':
                d = trial.suggest_int("d", 0, 2)
            else:
                d = 0

            model = fit_arima(engineered_train, model_configs={
                'p': trial.suggest_int("p", 0, 5),
                'd': d,
                'q': trial.suggest_int("q", 0, 5),
            })

            pred_df = predict_arima(model, num_predictions)

        elif method == 'prophet':
            engineered_train = engineered_train.reset_index()
            engineered_train = engineered_train.rename(
                columns={time_column: 'ds', engineered_train.columns[-1]: 'y'})
                
            model = fit_prophet(engineered_train, model_configs={
                "changepoint_prior_scale": trial.suggest_loguniform(
                    "changepoint_prior_scale", 0.001, 10.0
                ),
                "seasonality_prior_scale": trial.suggest_loguniform(
                    "seasonality_prior_scale", 0.01, 10.0
                ),
                "holidays_prior_scale": trial.suggest_loguniform(
                    "holidays_prior_scale", 0.01, 10.0
                ),
            })
            pred_df = predict_prophet(model, num_predictions)
        
        else:
            raise NotImplementedError()

        
        # Visualize forecast
        # Postprocess prediction
        no = seasonal_lag if fe_method == 'seasonal_diff' else 1
        last_indexes = norm_train.index[:no]
        last_values = norm_train.iloc[:no]
        predictions = postprocess(
            pred_df, sigma=sigma, mu=mu,
            last_indexes=last_indexes,
            last_values=last_values,
            cumsum=fe_method in ['diff', 'diff2', 'seasonal_diff'],
            cumsum_periods=seasonal_lag if fe_method == 'seasonal_diff' else 1,
        )

        # Evaluate the model
        score = validate(
            test_df[value_column].values, 
            predictions['predicted_mean'].loc[test_df.index].values
        )
        
        scores.append(score)

    avg_score = {}
    for metric in scores[0].keys():
        avg_score[metric] = get_mean_std([score[metric] for score in scores])

    return avg_score[best_key][0]

class Tuner:
    def __init__(
        self,
        storage:str=None,
        study_name: str = None,
        n_trials: int = 100,
        direction: str = "minimize",
        pruner=None,
        sampler=None,
        save_dir: str = None,
        use_best_params: bool = False,
    ) -> None:
        
        if storage is not None:
            if storage.endswith(".log"):
                self.storage = JournalStorage(JournalFileStorage(storage))
        else:
            self.storage = None

        self.save_dir = save_dir
        self.study_name = study_name
        self.n_trials = n_trials
        self.direction = direction
        self.pruner = pruner
        self.sampler = sampler
        self.save_dir = save_dir
        self.use_best_params = use_best_params
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage=self.storage,
            load_if_exists=True,
            pruner=pruner,
            sampler=sampler,
        )

    def tune(self, data, params):
        """Tune the model"""
        wrapped_objective = lambda trial: objective(
            trial, data, params
        )

        if not self.use_best_params:
            try:
                self.study.optimize(wrapped_objective, n_trials=self.n_trials)
            except KeyboardInterrupt:
                print("KeyboardInterrupt")
        best_trial = self.study.best_trial
        self.save_best_config(best_trial.params)

        if self.save_dir is not None:
            leaderboard_df = self.leaderboard()
            leaderboard_df.to_csv(osp.join(self.save_dir, 'leaderboard.csv'))
            leaderboard_df.to_json(osp.join(self.save_dir, 'leaderboard.json'), orient='records')
            print(f"Leaderboard saved to {self.save_dir}/leaderboard.csv")
            figs = self.visualize("all")
            os.makedirs(osp.join(self.save_dir, 'figures'), exist_ok=True)
            for fig_name, fig in figs:
                fig.write_image(osp.join(self.save_dir, 'figures', f"{fig_name}.png"))
                print(f"{fig_name} plot saved to {self.save_dir}/{fig_name}.png")

        return best_trial.params

    def save_best_config(self, best_params: Dict):
        if self.save_dir is None:
            return
        with open(os.path.join(self.save_dir, "best_config.json"), "w") as f:
            json.dump(best_params, f, indent=4)
        print(f"Best config saved to {self.save_dir}/best_config.json")

    def leaderboard(self):
        """Print leaderboard of all trials"""
        df = self.study.trials_dataframe()
        df.columns = [col.replace("user_attrs_", "") for col in df.columns]
        return df

    def visualize(self, plot: str, plot_params: dict = {}):
        """Visualize everything"""

        allow_plot_types = [
            "history",
            "contour",
            "edf",
            "intermediate_values",
            "parallel_coordinate",
            "param_importances",
            "slice",
        ]
        assert plot in ["all", *allow_plot_types], f"{plot} is not supported by Optuna"

        if plot == "history":
            fig = plot_optimization_history(self.study, **plot_params)
        elif plot == "contour":
            fig = plot_contour(self.study, **plot_params)
        elif plot == "edf":
            fig = plot_edf(self.study, **plot_params)
        elif plot == "intermediate_values":
            fig = plot_intermediate_values(self.study)
        elif plot == "parallel_coordinate":
            fig = plot_parallel_coordinate(self.study, **plot_params)
        elif plot == "param_importances":
            fig = plot_param_importances(self.study, **plot_params)
        elif plot == "slice":
            fig = plot_slice(self.study, **plot_params)
        elif plot == "all":
            fig = []
            for plot_type in allow_plot_types:
                one_fig = self.visualize(plot_type, plot_params)
                fig.append((plot_type, one_fig))
        else:
            print(f"{plot} is not supported by Optuna")
            raise ValueError()

        return fig