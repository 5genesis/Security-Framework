
import argparse
import logging
import time
import math

import pandas as pd
import tensorflow as tf
import numpy as np

from os import path
from math import sqrt
from numpy.random import seed
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.optimizers import SGD

from utils import getFeatures
from utils import getAnomalyColumns
from utils import loadDataset
from utils import saveNormalizationStats
from utils import saveDictJson
from utils import loadDictJson
from utils import normalizeFeature
from utils import split_sequences
from utils import plotMetric
from utils import plotAccLoss
from utils import createEmptyMetricsArray
from utils import convertNumpyToPandas
from utils import printPredictionErrors

from metrics_formatter import formatCpuSecondsTotal
from metrics_formatter import formatMemoryFreeBytes
from metrics_formatter import formatNetworkBytes
from metrics_formatter import format5gNetworkBytes
from metrics_formatter import format5gCpuPercentage

from influx_utils import initializeConnection
from influx_utils import checkDatabase
from influx_utils import insertAnomalies
from influx_utils import getLastRecords

import matplotlib.pyplot as plt

# Initialization values
seed(101)
tf.random.set_seed(seed=101)

pd.set_option('display.width', 1920)
pd.set_option('display.max_columns', 100)
pd.set_option('use_inf_as_na', True)

logging.basicConfig(
    filename='5g_anomaly_detection.log',
    format='%(asctime)s [%(levelname)-8s] %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.getLogger('matplotlib').setLevel(level=logging.CRITICAL)
logging.getLogger("requests").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("PIL.TiffImagePlugin").setLevel(logging.CRITICAL)

def evaluate(thresholds_file, cpu_testset, iperf_testset, trainset, time_window_threshold):
    plt.clf()

    stats_json = data_prefix + 'normalization_stats.json'

    model = load_model('model/5g_autoencoder.h5')
    normalization_stats = loadDictJson(stats_json)

    cols_to_normalize = getFeatures()
    cols = [c+'_normalized' for c in cols_to_normalize]

    time_window_threshold = 30
    refresh_time_interval = 15

    n_steps = 4
    n_features = len(cols)

    logging.info('Loading evaluation datasets')
    val_df = loadDataset(trainset)
    cpu_df = loadDataset(cpu_testset)
    iperf_df = loadDataset(iperf_testset)

    cpu_df.fillna(method='backfill', inplace=True)
    cpu_df.replace([np.inf, -np.inf], 0.0, inplace=True)

    iperf_df.fillna(method='backfill', inplace=True)
    iperf_df.replace([np.inf, -np.inf], 0.0, inplace=True)

    val_df.fillna(method='backfill', inplace=True)
    val_df.replace([np.inf, -np.inf], 0.0, inplace=True)

    logging.info('Normalizing evaluation data')
    for col in cols_to_normalize:
        cpu_df[col + '_normalized'] = normalizeFeature(cpu_df, col, normalization_stats[col + '_min'], normalization_stats[col + '_max'])
        iperf_df[col + '_normalized'] = normalizeFeature(iperf_df, col, normalization_stats[col + '_min'], normalization_stats[col + '_max'])
        val_df[col + '_normalized'] = normalizeFeature(val_df, col, normalization_stats[col + '_min'], normalization_stats[col + '_max'])

    logging.info('Evaluating for CPU and memory metrics')

    cpu_xs = []
    cpu_ys = []

    net_up_xs = []
    net_up_ys = []

    net_down_xs = []
    net_down_ys = []

    mem_xs_a1 = []
    mem_ys_a1 = []
    for sample_start in range(0, len(cpu_df)-time_window_threshold):
        sample_end = sample_start + time_window_threshold
        cpu_df_sample = cpu_df.iloc[sample_start:sample_end]

        # Select required columns for evaluation data batch
        cpu_dataset = cpu_df_sample[cols].to_numpy()

        # Prepare evaluation dataset batch
        X_test_cpu, y_test_cpu = split_sequences(cpu_dataset, n_steps)
        X_test_cpu = X_test_cpu.reshape((len(X_test_cpu), n_steps, n_features))

        # Predict for evaluation dataset batch
        yhat_cpu = model.predict(X_test_cpu, verbose=0)

        cpu_rmse_dict = printPredictionErrors(y_test_cpu, yhat_cpu)

        net_up_xs.append(len(net_up_xs))
        net_up_ys.append(cpu_rmse_dict['net_up_rmse'])

        net_down_xs.append(len(net_down_xs))
        net_down_ys.append(cpu_rmse_dict['net_down_rmse'])

        cpu_xs.append(len(cpu_xs))
        cpu_ys.append(cpu_rmse_dict['cpu_rmse'])

        mem_xs_a1.append(len(mem_xs_a1))
        mem_ys_a1.append(cpu_rmse_dict['mem_rmse'])

    plt.plot(cpu_xs, cpu_ys, color='blue', label='CPU Percentage Rate (mode=user)')
    #plt.plot(mem_xs_a1, mem_ys_a1, color='red', label='Memory Percentage Rate')
    plt.title('CPU Attack Dataset')
    plt.xlabel('# of Sequence')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig('plots/evaluate_cpu.png')
    plt.clf()
    
    logging.info('Evaluating for network and 5G metrics')

    net_up_xs = []
    net_up_ys = []

    net_down_xs = []
    net_down_ys = []

    net_5g_up_xs = []
    net_5g_up_ys = []

    net_5g_down_xs = []
    net_5g_down_ys = []

    mem_xs_a2 = []
    mem_ys_a2 = []
    for sample_start in range(0, len(iperf_df)-time_window_threshold):
        sample_end = sample_start + time_window_threshold
        iperf_df_sample = iperf_df.iloc[sample_start:sample_end]

        # Select required columns for evaluation data batch
        iperf_dataset = iperf_df_sample[cols].to_numpy()

        # Prepare evaluation dataset batch
        X_test_iperf, y_test_iperf = split_sequences(iperf_dataset, n_steps)
        X_test_iperf = X_test_iperf.reshape((len(X_test_iperf), n_steps, n_features))

        # Predict for evaluation dataset batch
        yhat_iperf = model.predict(X_test_iperf, verbose=0)

        iperf_rmse_dict = printPredictionErrors(y_test_iperf, yhat_iperf)

        net_up_xs.append(len(net_up_xs))
        net_up_ys.append(iperf_rmse_dict['net_up_rmse'])

        net_down_xs.append(len(net_down_xs))
        net_down_ys.append(iperf_rmse_dict['net_down_rmse'])

        net_5g_up_xs.append(len(net_5g_up_xs))
        net_5g_up_ys.append(iperf_rmse_dict['net_up_5g_rmse'])

        net_5g_down_xs.append(len(net_5g_down_xs))
        net_5g_down_ys.append(iperf_rmse_dict['net_down_5g_rmse'])

        mem_xs_a2.append(len(mem_xs_a2))
        mem_ys_a2.append(iperf_rmse_dict['mem_rmse'])
    
    plt.plot(net_up_xs, net_up_ys, color='green', label='Network Up Rate')
    plt.plot(net_down_xs, net_down_ys, color='purple', label='Network Down Rate')
    #plt.plot(mem_xs_a2, mem_ys_a2, color='red', label='Memory Percentage Rate')
    plt.title('iperf Attack Dataset')
    plt.xlabel('# of Sequence')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig('plots/evaluate_iperf_net.png')
    plt.clf()

    plt.plot(net_5g_up_xs, net_5g_up_ys, color='green', label='5G Network Up Rate')
    plt.plot(net_5g_down_xs, net_5g_down_ys, color='blue', label='5G Network Down Rate')
    plt.title('iperf Attack Dataset')
    plt.xlabel('# of Sequence')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig('plots/evaluate_iperf_5g.png')
    plt.clf()
    
    logging.info('Evaluating with training data')

    cpu_xs = []
    cpu_ys = []

    net_up_xs = []
    net_up_ys = []

    net_down_xs = []
    net_down_ys = []

    net_5g_up_xs = []
    net_5g_up_ys = []

    net_5g_down_xs = []
    net_5g_down_ys = []

    mem_xs_n = []
    mem_ys_n = []
    for sample_start in range(0, len(val_df)-time_window_threshold):
        sample_end = sample_start + time_window_threshold
        val_df_sample = val_df.iloc[sample_start:sample_end]

        # Select required columns for evaluation data batch
        val_dataset = val_df_sample[cols].to_numpy()

        # Prepare evaluation dataset batch
        X_test_val, y_test_val = split_sequences(val_dataset, n_steps)
        X_test_val = X_test_val.reshape((len(X_test_val), n_steps, n_features))

        # Predict for evaluation dataset batch
        yhat_val = model.predict(X_test_val, verbose=0)

        val_rmse_dict = printPredictionErrors(y_test_val, yhat_val)

        cpu_xs.append(len(cpu_xs))
        cpu_ys.append(val_rmse_dict['cpu_rmse'])

        mem_xs_n.append(len(mem_xs_n))
        mem_ys_n.append(val_rmse_dict['mem_rmse'])

        net_up_xs.append(len(net_up_xs))
        net_up_ys.append(val_rmse_dict['net_up_rmse'])

        net_down_xs.append(len(net_down_xs))
        net_down_ys.append(val_rmse_dict['net_down_rmse'])

        net_5g_up_xs.append(len(net_5g_up_xs))
        net_5g_up_ys.append(val_rmse_dict['net_up_5g_rmse'])

        net_5g_down_xs.append(len(net_5g_down_xs))
        net_5g_down_ys.append(val_rmse_dict['net_down_5g_rmse'])

    plt.plot(cpu_xs, cpu_ys, color='blue', label='CPU Percentage Rate (mode=user)')
    plt.plot(mem_xs_n, mem_ys_n, color='red', label='Memory Percentage Rate')
    plt.plot(net_up_xs, net_up_ys, color='green', label='Network Up Rate')
    plt.plot(net_down_xs, net_down_ys, color='purple', label='Network Down Rate')
    plt.title('Training Dataset (Edge Metrics)')
    plt.xlabel('# of Sequence')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig('plots/evaluate_val_1.png')
    plt.clf()

    plt.plot(net_5g_up_xs, net_5g_up_ys, color='orange', label='5G Network Up Rate')
    plt.plot(net_5g_down_xs, net_5g_down_ys, color='cyan', label='5G Network Down Rate')
    plt.title('Training Dataset (5G Metrics)')
    plt.xlabel('# of Sequence')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig('plots/evaluate_val_2.png')
    plt.clf()


if __name__ == "__main__":
    data_prefix = 'data/'
    model_prefix = 'model/'
    plots_prefix = 'plots/'

    trainset = data_prefix + 'normal.csv' 
    cpu_testset = data_prefix + 'cpu_attack.csv' 
    iperf_testset = data_prefix + 'iperf_attack.csv' 

    time_window_threshold = 30

    evaluate(None, cpu_testset, iperf_testset, trainset, time_window_threshold)