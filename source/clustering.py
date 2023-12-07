import numpy as np
import pandas as pd
import os
import os.path as osp
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from tsfresh import extract_features, select_features
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from sklearn.decomposition import PCA



features = extract_features(merged_df, column_id='id', column_sort='date', column_value='value')
features.head()