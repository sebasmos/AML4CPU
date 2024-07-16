import pandas as pd
import numpy as np
import os
from pathlib import Path
import copy
from pandas import DataFrame, concat

from sklearn.preprocessing import StandardScaler

DATASET_PATH = Path(__file__).parent / "./data"

DATA = f"{DATASET_PATH}/cpu_data_custom.csv"

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return 200 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))


def denormalize_predictions_std_mean(predictions, mean_y, std_y):
    return (predictions * std_y) + mean_y

def denormalize_predictions(y_pred_normalized, normalization, scaler_y=None, y_min=0, y_max=None):
    if normalization == 0:
        y_pred = y_pred_normalized
    elif normalization == 1:
        y_pred_normalized = y_pred_normalized.reshape(1, -1)  # [1,X]
        y_pred = scaler_y.inverse_transform(y_pred_normalized)
        y_pred = y_pred.squeeze(0)  # [X,]
    elif normalization == 2:
        """
        Denormalization 2 implies to use min-max normalization values used during training - assuming same logic as used for scikit-learn scaler
        instead of using the min-max values from the testing set itself.
        """
        y_pred = y_min + y_pred_normalized * (y_max - y_min)
    return y_pred

def normalize_data(X_train, X_test, y_train, y_test):
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=X_test.columns if isinstance(X_test, pd.DataFrame) else None)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=X_train.columns if isinstance(X_train, pd.DataFrame) else None)
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train, name=y_test.name if isinstance(y_test, pd.Series) else None)
    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test, name=y_train.name if isinstance(y_train, pd.Series) else None)

    scaler_X = StandardScaler()
    scaler_X.fit(X_train)
    X_train_normalized = pd.DataFrame(scaler_X.transform(X_train), columns=X_train.columns)
    X_test_normalized = pd.DataFrame(scaler_X.transform(X_test), columns=X_test.columns)

    scaler_y = StandardScaler()
    scaler_y.fit(y_train.values.reshape(-1, 1))
    y_train_normalized = pd.Series(scaler_y.transform(y_train.values.reshape(-1, 1)).flatten(), name=y_train.name)
    y_test_normalized = pd.Series(scaler_y.transform(y_test.values.reshape(-1, 1)).flatten(), name=y_test.name)

    return X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized, scaler_y

def mimax_norm(x, min, max):
    return (x-min)/(max-min)

def load_data(data_path, small_sample=False): 
    
    if small_sample:
        train_data_path = f"{data_path}/train_data_small.csv"
        test_data_path = f"{data_path}/test_data_small.csv"
    else:
        train_data_path = f"{data_path}/train_data.csv"
        test_data_path = f"{data_path}/test_data.csv"
    data_train = pd.read_csv(train_data_path, index_col=0)
    data_test = pd.read_csv(test_data_path, index_col=0)

    return data_train, data_test

    return data_train, data_test
def load_orangepi_data(data_path="./data/cpu_data_custom_adapted.csv"):
    """
    Loads the training and test data from the specified files.
    
    Returns:
        data_train (pd.DataFrame): The training data.
        data_test (pd.DataFrame): The test data.
    """
    df = pd.read_csv(data_path)

    df = copy.deepcopy(df)

    df = df[['Date', 'CPU Core 1 Usage (%)', 'CPU Core 2 Usage (%)']]

    cols = {'Date': 'Date', 'CPU Core 1 Usage (%)': 'TARGET', 'CPU Core 2 Usage (%)': 'RAM'}
    df = df.rename(columns=cols, inplace=False)

    return df

def load_orangepi_data_per_hour():
    """
    Loads the training and test data from the specified files.
    
    Returns:
        data_train (pd.DataFrame): The training data.
        data_test (pd.DataFrame): The test data.
    """
    df = pd.read_csv(DATA)

    df = copy.deepcopy(df)

    df = df[['Date', 'CPU Core 1 Usage (%)', 'CPU Core 2 Usage (%)']]

    cols = {'Date': 'date', 'CPU Core 1 Usage (%)': 'TARGET', 'CPU Core 2 Usage (%)': 'RAM'}
    df = df.rename(columns=cols, inplace=False)
    df['timestamp'] = pd.to_datetime(df['date'])
    df.set_index('timestamp', inplace=True)
    resampled_df = df.resample('1H').mean()
    sampled_df = resampled_df.sample(n=45, random_state=42)
    
    return sampled_df
    
def ts_supervised_structure(data, n_in=1, n_out=1, dropnan=True, autoregressive=True):
    no_autoregressive = not(autoregressive)
    if no_autoregressive:
        n_in = n_in - 1

    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        if no_autoregressive:
            cols.append(df.shift(i).iloc[:,:-1])
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars-1)]
        else:
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
            
def structure(data, window_size = 5):
    for k in range(1, window_size + 1):
                    data[f'cpu_lag{k}'] = data['TARGET'].shift(k)
    data.dropna(inplace=True)
    X_train = data[[f'cpu_lag{k}' for k in range(1, window_size + 1)]]
    y_train = data['TARGET']
    y_train.reset_index(drop=True)
    X_train.reset_index(drop=True)
    return X_train, y_train

def normalize_data_std(X):
    X_normalized = (X - np.mean(X)) / np.std(X)
    return X_normalized

def normalize_data(X_train, X_test, y_train, y_test):
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=X_test.columns if isinstance(X_test, pd.DataFrame) else None)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=X_train.columns if isinstance(X_train, pd.DataFrame) else None)
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train, name=y_test.name if isinstance(y_test, pd.Series) else None)
    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test, name=y_train.name if isinstance(y_train, pd.Series) else None)

    # Normalize X data
    scaler_X = StandardScaler()
    scaler_X.fit(X_train)
    X_train_normalized = pd.DataFrame(scaler_X.transform(X_train), columns=X_train.columns)
    X_test_normalized = pd.DataFrame(scaler_X.transform(X_test), columns=X_test.columns)

    # Normalize y data
    scaler_y = StandardScaler()
    scaler_y.fit(y_train.values.reshape(-1, 1))
    y_train_normalized = pd.Series(scaler_y.transform(y_train.values.reshape(-1, 1)).flatten(), name=y_train.name)
    y_test_normalized = pd.Series(scaler_y.transform(y_test.values.reshape(-1, 1)).flatten(), name=y_test.name)

    return X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized, scaler_y
