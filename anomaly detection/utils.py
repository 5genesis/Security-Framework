"""
Helper Functions
"""

import json
import math
import logging
import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from math import sqrt
from numpy import array
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.DEBUG)

logging.getLogger('matplotlib').setLevel(level=logging.CRITICAL)
logging.getLogger("requests").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

def loadDataset(dataset):
    """
    Loads a dataset into a Pandas DataFrame. Dataset file must be in .csv format. 
    It must be separated with commas (,), has a header with column names. 
    Also, it must has a column named time, which will have timestamp values.
    Column time is parsed as date and set as the index of the generated DataFrame.

    param dataset: File where the dataset is saved.

    return: Dataframe generated from dataset param
    """

    df = pd.read_csv(dataset, header=0, delimiter=',', index_col='time', parse_dates=['time'])
    df.index = pd.to_datetime(df.index)

    return df

def saveDictJson(dict, filename):
    """
    Dumps a Python dict object to a given file. Created file will be in JSON format.

    param dict: Python dict object.
    param filename: Filename(included path) where the dict will be saved.

    return: None
    """

    with open(filename, 'w') as fp:
      json.dump(dict, fp)

def loadDictJson(filename):
    """
    Loads a Python dict object from a given file. Given file must be in JSON format.

    param filename: Filename(included path) that will be converted in dict.

    return: The loaded Python dict object.
    """

    d = {}
    with open(filename) as fp:
      d = json.load(fp)

    return d

def saveNormalizationStats(df, cols_to_normalize):
    """
    Dump a Python dict, that contains min & max value for each feature, into a JSON file.
    These min & max values will be used to normalize required features.

    param df: Pandas DataFrame object
    param cols_to_normalize: List of columns to normalize from df param

    return: A Python dict object, that contains min & max values for each feature that will be normalized
    """

    normalization_stats = {}

    for col in cols_to_normalize:
        normalization_stats[col + '_min'] = df[col].min()
        normalization_stats[col + '_max'] = df[col].max()

    return normalization_stats

# Get cols from data, that will be used for training or testing
def getFeatures():
    """
    Creates a list of columns ftom dataset, that will be used for training & testing

    return: List of columns from dataset
    """

    return [
        'cpu_user_rate', 'cpu_rx_rate', 'cpu_tx_rate', 
        'bytes_received_rate', 'bytes_transmitted_rate',
        'dl_bytes_rate', 'ul_bytes_rate',
        'mem_rate'
    ]

# Get columns for detected anomalies to save in InfluxDB
def getAnomalyColumns():
    """
    Creates a list of columns from dataset, that will be saved in InfluxDB when an anomaly is detected

    return: List of columns from dataset
    """

    return [
        'cpu_user_percentage', 'mem_percentage', 'bytes_received', 'bytes_transmitted',
        'cpu_rx_percentage', 'cpu_tx_percentage', 'dl_bytes', 'ul_bytes'
    ]

def getColumnNames():
    """ 
    Creates a Python dict object, where dict's keys are columns names and dict's values are columns' types.

    return: dict with columns names and types.
    """

    df_columns = {
        'cpu_user_0_percentage': float,
        'cpu_user_0_rate': float,
        'cpu_user_1_percentage': float,
        'cpu_user_1_rate': float,
        'enp1s0_bytes_received': float,
        'enp1s0_received_rate': float,
        'enp0s20u1_bytes_received': float,
        'enp0s20u1_received_rate': float,
        'ppp0_bytes_received': float,
        'ppp0_received_rate': float,
        'enp1s0_bytes_transmitted': float,
        'enp1s0_transmitted_rate': float,
        'enp0s20u1_bytes_transmitted': float,
        'enp0s20u1_transmitted_rate': float,
        'ppp0_bytes_transmitted': float,
        'ppp0_transmitted_rate': float,
        'mem': float,
        'mem_percentage': float,
        'mem_rate': float,
        'cpu_user_percentage': float,
        'cpu_user_rate': float,
        'bytes_received': float,
        'bytes_received_rate': float,
        'bytes_transmitted': float,
        'bytes_transmitted_rate': float,
        'ul_bytes': float,
        'ul_bytes_rate': float,
        'dl_bytes': float,
        'dl_bytes_rate': float,
        'cpu_rx_percentage': float,
        'cpu_rx_rate': float,
        'cpu_tx_percentage': float,
        'cpu_tx_rate': float
    }

    return df_columns

def normalizeFeature(df, feature, min, max):
    """
    Min-Max Normalization. Creates a new Pandas Series with normalized values of another Pandas Series. 
    Minimum and Maximum values that will be used for normalization must be provided.

    param df: Pandas DataFrame object
    param feature: Column of df param, that will be normalized
    param min: Minimum value that will be used for normalization
    param max: maximum value that will be used for normalization

    return: Pandas Series, where its values is the normalized ones from feature param according to min and max params
    """

    normalized = (df[feature].to_numpy() - min) / (max - min)
    return normalized

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    """
    Split a multivariate sequence into samples. (n-1) steps will be considered as previous records in the sequence, 
    and last step will be the current value of the sequence. The first (n-1) will be used from DL model in order to
    predict the n-th value of the sequence.

    param sequnces: Sequence that will be splited.
    param n_steps: Number of steps, that will be used for produced splitted sequences.

    return: array(X),array(y)
    array(X): Array containing the previous values of the sequence
    array(y): Array containing the current value of the sequence
    """

    X, y = list(), list()
    for i in range(len(sequences)):
		# find the end of this pattern
        end_ix = i + n_steps
		# check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break
		# gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

def plotMetric(history, metric, plots_prefix):
    """
    Plots metric and save it to .eps and .png files

    param history: history of training
    param metric: metric name
    param plots_prefix: data path, where plots are saved

    return: None
    """

    plt.clf()

    plt.plot(history.history[str(metric)])
    plt.plot(history.history['val_' + str(metric)])
    plt.title('5G LSTMs Anomaly Detector ' + str(metric))
    plt.ylabel(str(metric))
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    # Save plots
    plt.savefig(plots_prefix + str(metric) + '_plot.eps')
    plt.savefig(plots_prefix + str(metric) + '_plot.png')

def plotAccLoss(history, plots_prefix):
    """
    Plots Accuracy & Loss metrics in one plot together
    and save it to .eps, .png and .tiff file

    param history: history of training
    param plots_prefix: data path, where plots are saved

    return: None
    """

    plt.clf()

    plt.rcParams.update({'font.size': 12})
    plt.rcParams.update({'font.weight': 'bold'})

    fig, axs = plt.subplots(2)

    fig.suptitle('5G LSTMs Anomaly Detector Accuracy & Loss', fontdict=dict(weight='bold'), fontsize=12)

    axs[0].plot(history.history['acc'])
    axs[0].plot(history.history['val_acc'])
    axs[0].set_xlabel('Epoch', fontdict=dict(weight='bold'), fontsize=12)
    axs[0].set_ylabel('Accuracy', fontdict=dict(weight='bold'), fontsize=12)
    axs[0].legend(['train', 'validation'], loc='lower right')

    axs[1].plot(history.history['loss'])
    axs[1].plot(history.history['val_loss'])
    axs[1].set_xlabel('Epoch', fontdict=dict(weight='bold'), fontsize=12)
    axs[1].set_ylabel('Loss', fontdict=dict(weight='bold'), fontsize=12)
    axs[1].legend(['train', 'validation'], loc='upper right')

    # Save plot
    plt.savefig(plots_prefix + 'acc_loss_plot_300.eps', dpi=300)
    plt.savefig(plots_prefix + 'acc_loss_plot.eps')
    plt.savefig(plots_prefix + 'acc_loss_plot_300.png', dpi=300)
    plt.savefig(plots_prefix + 'acc_loss_plot.png')
    plt.savefig(plots_prefix + 'acc_loss_plot_300.tiff', dpi=300)
    plt.savefig(plots_prefix + 'acc_loss_plot.tiff')

def createEmptyMetricsArray(col_num):
    """
    Creates an empty numpy array with one row and col_num columns, filled with zeros.

    param col_num: Number of columns.

    return: An array with size (1 x col_num) filled with zeros.
    """

    metrics_array = np.zeros(shape=(1,col_num))
    metrics_array[0,0] = float(time.time())

    return metrics_array

def convertNumpyToPandas(np_arr):
    """
    Create a Pandas DataFrame object from a numpy array.
    First column of numpy array will be used as index.

    param np_arr: numpy array to be converted in Pandas DataFrame
    
    return: The ccreated Pandas DataFrame
    """

    df = pd.DataFrame(data=np_arr[0:,1:], index=np_arr[0:,0], columns=getColumnNames())
    
    return df

def printPredictionErrors(y_actual, y_predict):
    """
    Calculate RMSE for predicted features.
    Creates a dict with calculate RMSE for all features combined, and for each one separately.

    param y_actual: Real values of a sequence.
    param y_predict: Predicted values of a sequence.

    return: Dict with calculated RMSEs.
    """

    rmse_dict = {}
    
    # Calculate total RMSE
    rmse_dict['rmse_total'] = mean_squared_error(y_actual, y_predict, squared=False)

    # Calculate RMSE for each feature
    rmse_dict['cpu_rmse'] = sqrt(mean_squared_error(y_actual[:,0], y_predict[:,0]))
    rmse_dict['cpu_rx_rmse'] = sqrt(mean_squared_error(y_actual[:,1], y_predict[:,1]))
    rmse_dict['cpu_tx_rmse'] = sqrt(mean_squared_error(y_actual[:,2], y_predict[:,2]))
    rmse_dict['net_down_rmse'] = sqrt(mean_squared_error(y_actual[:,3], y_predict[:,3]))
    rmse_dict['net_up_rmse'] = sqrt(mean_squared_error(y_actual[:,4], y_predict[:,4]))
    rmse_dict['net_down_5g_rmse'] = sqrt(mean_squared_error(y_actual[:,5], y_predict[:,5]))
    rmse_dict['net_up_5g_rmse'] = sqrt(mean_squared_error(y_actual[:,6], y_predict[:,6]))
    rmse_dict['mem_rmse'] = sqrt(mean_squared_error(y_actual[:,7], y_predict[:,7]))

    return rmse_dict