"""
Main Module

Program Execution Params:
param --mode: Mode to run the algorithm. (Available: Train/Test)
param --model: Filename to save/load the trained model.
param --evaluate: Run evaluation test in order to propose missing thresholds. 
                  If all thresholds are set by user, evaluation is not required. 
                  In case that some thresholds are not set and evaluation is 
                  disabled, non-set thresholds will be set with default values.
param --thresholds_file: Filename, where thresholds are saved/loaded in JSON format.
param --influx_host: Host IP of InfluxDB.
param --influx_port: Port InfluxDB is listening.
param --influx_user: InfluxDB username.
param --influx_pass: InfluxDB user's password.
param --influx_db: InfluxDB database, where metrics and anomalies are stored.
param --influx_measurement: InfluxDB measurement in influx_db where anomalies are stored.


Mode: Train
Deep Learning model must be trained with a dataset containing normal records. 
Training data have been collected from both Prometheus and Amarisoft exporter.
----- Preprocessing ----- 
Because of these two different systems there is a small difference in timestamps
for these metrics. For this reason the training dataset is resampled every 
15 seconds, to synchronize records from these two systems. 
For training the following features have been selected: CPU percentage rate, 
RAM percentage rate, RX/TX CPU percentage rate, rate of transmitted/received 
bytes and rate of bytes downloaded/uploaded from 5G interface.
These features will be used normalized using Min-Max Normalization. The min & max
values that will be used for normalization will be saved in a JSON file. After
resampling and normalization the dataset is splitted in sequences, based on a 
number of steps parameter. This parameters is set to 4, which means that for each 
bunch of 4 records, the first 3 will be used as historical data for the algorithm, 
and the fourth is the one that must be predicted.
----- Training ----- 
For training the architecture of an autoencoder has been used. Model consisted of
9 Biderectional LSTM layers and one Dense layer at the end. As activation function
ReLu is used. SGD with Nesterov momentum has been used as optimizer, with learning 
rate equal to 0.01. A part of the trainset will be used as validation set. By default 
it is 10% of the training set.
----- Evaluation ----- 
If evaluation is enabled, trained model will be loaded and used for testing in 
three different datasets. The first dataset has been collected during a CPU 
stress test attack. The second dataset has been collected during an iperf stress
test attack. The third dataset is the training set. All these datasets will be 
normalized with saved normalization values during training. Using these datasets 
the RMSE error between the actual and predicted values will be calculated. 
Network features' thresholds, will be not affected from CPU stress dataset, and 
CPU features' thresholds, will be not affected from iperf stress dataset. The 
99th percentile of each feature's RMSE will be considered as threshold from 
anomaly detection algorithm. The 99th percentile has been selected compared with 
the max value, in order to avoid outliers, that may exist in these datasets. 
User-defined thresholds will not be updated from calculated RMSEs.


Mode: Test
Trained Model will be used in order to predict the next features' values in the 
time series. Because it is trained in normal traffic it will predict these values 
considering that the incoming traffic is normal. If the RMSE between the predicted 
and actual value is above a set threshold this record will be considered as anomaly. 
Algorithm fetches the last record from an InfluxDB every 15 seconds. For predicting 
the next value a sliding time window has been used, keeping only the last n records. 
By default this n is set to be equal to 30. Fetched records are been preprocessed 
and normalized before used by the model for predicting. After prediction the RMSes 
between the actual and predicted values are calculated and compared with set thresholds, 
to identify if the record considers an anomaly or not. A possible cause of the anomaly 
is also saved in the database with the detected anomalies or a 'unknown cause' message 
if the cause cannot be identified.
"""

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

data_prefix = 'data/'
model_prefix = 'model/'
plots_prefix = 'plots/'

trainset = data_prefix + 'normal.csv' 
cpu_testset = data_prefix + 'cpu_attack.csv' 
iperf_testset = data_prefix + 'iperf_attack.csv' 

stats_json = data_prefix + 'normalization_stats.json'

def train(train_dataset, n_steps, n_features, train_epochs=10, val_split=0.1, train_verbose=1, model_filename='model/5g_edge_autoencoder.h5'):
    """
    Train DL model.

    param train_dataset: Dataset, that will be used for training model.
    param n_steps: How many previous steps in sequence will be used for training.
    param n_features: Number of features that will be used for training.
    param train_epochs: Epochs to train the model. (Default: 10)
    param val_split: Percentage of training dataset to use as validation dataset. (Values: 0.0 - 1.0) (Default: 0.1)
    param train_verbose: Show output of training. (Values: 0, 1, 2) (Default: 1)
    param model_filename: Filename to save trained model. (Default: model/5g_edge_autoencoder.h5)

    return: Trained model.
    """

    logging.info('Converting train & validation data to fit model')
    # Convert into input/output
    X, y = split_sequences(train_dataset, n_steps)
    logging.info('Input Shape: ' + str(X.shape))
    logging.info('Output Shape: ' + str(y.shape))
    
    # Model Architecture
    model = Sequential()
    model.add(Bidirectional(LSTM(64, activation='relu', return_sequences=True, input_shape=(n_steps, n_features))))
    model.add(Bidirectional(LSTM(32, activation='relu', return_sequences=True)))
    model.add(Bidirectional(LSTM(16, activation='relu', return_sequences=True)))
    model.add(Bidirectional(LSTM(8, activation='relu', return_sequences=True)))
    model.add(Bidirectional(LSTM(4, activation='relu', return_sequences=True)))
    model.add(Bidirectional(LSTM(8, activation='relu', return_sequences=True)))
    model.add(Bidirectional(LSTM(16, activation='relu', return_sequences=True)))
    model.add(Bidirectional(LSTM(32, activation='relu', return_sequences=True)))
    model.add(Bidirectional(LSTM(64, activation='relu')))
    model.add(Dense(n_features))

    model.compile(
        optimizer=SGD(learning_rate=0.01, nesterov=True), 
        loss='mse',
        metrics=['acc']
    )

    # Train model
    logging.info('Training model')
    history = model.fit(
        X, y, 
        epochs=train_epochs, 
        validation_split=val_split, 
        verbose=1
    )
    
    logging.info(model.summary())

    logging.info('Saving trained model to file ' + model_filename)
    model.save(model_filename)

    logging.info('Saving trained model in json format in file ' + model_prefix + '5g_autoencoder.json')
    with open(model_prefix + '5g_autoencoder.json', "w") as file:
        file.write(model.to_json())
    
    logging.info('Saving model weights in file ' + model_prefix + '5g_autoencoder_weights.h5')
    model.save_weights(model_prefix + '5g_autoencoder_weights.h5')

    plotMetric(history, 'loss', plots_prefix)
    plotMetric(history, 'acc', plots_prefix)
    plotAccLoss(history, plots_prefix)

    return model

def evaluate(thresholds_file, cpu_testset, iperf_testset, trainset, time_window_threshold):
    """
    Evaluate trained model. If user has not set all thresholds for anomalies, evaluation will also set
    the remaining thresholds. Evaluation uses a dataset, that contains a CPU stress test, an iperf 
    stress test and predicting the data used for training. Thresholds are defined by calculating the 
    RMSEs from actual values and taking the 99th percentile of these errors for each feature separately and overall.
    If user has set all thresholds when starting the program,these thresholds will be used.

    param thresholds_file: File, where user-defined thresholds are saved. This will updated if new thresholds are proposed.
    param cpu_testset: File containing dataset with CPU stress test.
    param iperf_testset: File containing dataset with iperf stress test.
    param trainset: File containing the dataset used for training.
    param time_window_threshold: Time window for keeping the last-n records. In evaluation data are predicted in batches of n.

    return: None.
    """
    

    # Loading thresholds from file. Create an empty dict if no file exists
    thresholds_dict = {}
    if path.exists(thresholds_file):
        thresholds_dict = loadDictJson(thresholds_file)
    

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

    cpu_rmse = []
    cpu_rx_rmse = []
    cpu_tx_rmse = []
    net_down_rmse = []
    net_up_rmse = []
    net_down_5g_rmse = []
    net_up_5g_rmse = []
    mem_rmse = []
    total_rmse = []

    logging.info('Evaluating for CPU and memory metrics')

    sequences = []
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

        sequences.append(len(sequences))

        total_rmse.append(cpu_rmse_dict['rmse_total'])
        cpu_rmse.append(cpu_rmse_dict['cpu_rmse'])
        cpu_rx_rmse.append(cpu_rmse_dict['cpu_rx_rmse'])
        cpu_tx_rmse.append(cpu_rmse_dict['cpu_tx_rmse'])
        mem_rmse.append(cpu_rmse_dict['mem_rmse'])

    logging.info('Evaluating for network and 5G metrics')

    sequences = []
    for sample_start in range(0, len(iperf_df)-time_window_threshold):
        sample_end = sample_start + time_window_threshold
        iperf_df_sample = iperf_df.iloc[sample_start:sample_end]

        # Select required columns for evaluation data batch
        iperf_dataset = iperf_df[cols].to_numpy()

        # Prepare evaluation dataset batch
        X_test_iperf, y_test_iperf = split_sequences(iperf_dataset, n_steps)
        X_test_iperf = X_test_iperf.reshape((len(X_test_iperf), n_steps, n_features))

        # Predict for evaluation dataset batch
        yhat_iperf = model.predict(X_test_iperf, verbose=0)

        iperf_rmse_dict = printPredictionErrors(y_test_iperf, yhat_iperf)

        sequences.append(len(sequences))

        total_rmse.append(iperf_rmse_dict['rmse_total'])
        net_down_rmse.append(iperf_rmse_dict['net_down_rmse'])
        net_up_rmse.append(iperf_rmse_dict['net_up_rmse'])
        net_down_5g_rmse.append(iperf_rmse_dict['net_down_5g_rmse'])
        net_up_5g_rmse.append(iperf_rmse_dict['net_up_5g_rmse'])
        mem_rmse.append(iperf_rmse_dict['mem_rmse'])

    logging.info('Evaluating with training data')

    sequences = []
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

        sequences.append(len(sequences))

        total_rmse.append(val_rmse_dict['rmse_total'])
        cpu_rmse.append(val_rmse_dict['cpu_rmse'])
        cpu_rx_rmse.append(val_rmse_dict['cpu_rx_rmse'])
        cpu_tx_rmse.append(val_rmse_dict['cpu_tx_rmse'])
        mem_rmse.append(val_rmse_dict['mem_rmse'])
        net_down_rmse.append(val_rmse_dict['net_down_rmse'])
        net_up_rmse.append(val_rmse_dict['net_up_rmse'])
        net_down_5g_rmse.append(val_rmse_dict['net_down_5g_rmse'])
        net_up_5g_rmse.append(val_rmse_dict['net_up_5g_rmse'])

    # For thresholds, that are not defined by user, use suggested values
    if 'cpu_threshold' not in thresholds_dict.keys():
        thresholds_dict['cpu_threshold'] = np.percentile(cpu_rmse, 0.99)
    if 'mem_threshold' not in thresholds_dict.keys():
        thresholds_dict['mem_threshold'] = np.percentile(mem_rmse, 0.99)
    if 'cpu_tx_threshold' not in thresholds_dict.keys():
        thresholds_dict['cpu_tx_threshold'] = np.percentile(cpu_tx_rmse, 0.99)
    if 'cpu_rx_threshold' not in thresholds_dict.keys():
        thresholds_dict['cpu_rx_threshold'] = np.percentile(cpu_rx_rmse, 0.99)
    if 'net_up_threshold' not in thresholds_dict.keys():
        thresholds_dict['net_up_threshold'] = np.percentile(net_up_rmse, 0.99)
    if 'net_down_threshold' not in thresholds_dict.keys():
        thresholds_dict['net_down_threshold'] = np.percentile(net_down_rmse, 0.99)
    if 'net_5g_up_threshold' not in thresholds_dict.keys():
        thresholds_dict['net_5g_up_threshold'] = np.percentile(net_up_5g_rmse, 0.99)
    if 'net_5g_down_threshold' not in thresholds_dict.keys():
        thresholds_dict['net_5g_down_threshold'] = np.percentile(net_down_5g_rmse, 0.99)
    if 'overall_threshold' not in thresholds_dict.keys():
        thresholds_dict['overall_threshold'] = np.percentile(total_rmse, 0.99)

    # Save new thresholds in same file
    saveDictJson(thresholds_dict, thresholds_file)

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, help='Train/Test.')
    parser.add_argument('--model', required=True, help='Model filename to save/load.')
    parser.add_argument('--evaluate', required=False, nargs='?', help='Test trained model with 2 different attacks (CPU overload, iperf stresstest) and training dataset. Suggest some thresholds. Default: False')
    parser.add_argument('--thresholds_file' , required=True, help='File with thresholds for anomaly detection')
    parser.add_argument('--influx_host', required=False, help='InfluxDB Host IP')
    parser.add_argument('--influx_port', required=False, help='InfluxDB Host Port')
    parser.add_argument('--influx_user', required=False, help='InfluxDB Username')
    parser.add_argument('--influx_pass', required=False, help='InfluxDB Password')
    parser.add_argument('--influx_db', required=False, help='InfluxDB Database')
    parser.add_argument('--influx_measurement', required=False, help='InfluxDB Measurement')

    args = parser.parse_args()

    cols_to_normalize = getFeatures()
    cols = [c+'_normalized' for c in cols_to_normalize]

    # Number of time steps
    n_steps = 4
    # Number of features, that will be used
    n_features = len(cols)

    # Number of max historicaal records to keep for predicting
    time_window_threshold = 30

    refresh_time_interval = 15

    if args.mode == 'train':
        logging.info('Mode: Training')
        logging.info('Evaluation: ' + str(True if args.evaluate in ('True', 'true') else False))
    
        # Initialization of training params
        # Time interval to resample data
        sample_time = '60S'

        logging.info('Loading train & validation dataset')
        df = loadDataset(trainset)

        # Resample every 15 sec, because prometheus & amari exporter have different timestamps
        df = df.resample('15S', closed='right', label='left').mean()

        # Fill train df with 0s if there are NaNs and Ifns
        df.fillna(0, inplace=True)

        logging.info('Saving normalization values')
        normalization_stats = saveNormalizationStats(df, cols_to_normalize)

        saveDictJson(normalization_stats, stats_json)

        logging.info('Normalizing train & validation data')
        for col in cols_to_normalize:
            df[col + '_normalized'] = normalizeFeature(df, col, normalization_stats[col + '_min'], normalization_stats[col + '_max'])

        logging.info('Selecting required columns for train & validation data')
        train_dataset = df[cols].to_numpy()
        
        start_time = time.time()
        model = train(train_dataset, n_steps, n_features, train_epochs=50, val_split=0.1, train_verbose=1, model_filename=args.model)
        elapsed_time = int(time.time() - start_time)

        logging.info('Training completed!')
        logging.info('Time elapsed for training: ' + str(elapsed_time) + ' seconds')
        
        if args.evaluate in ('True', 'true'):
            evaluate(args.thresholds_file, cpu_testset, iperf_testset, trainset, time_window_threshold)
            logging.info('Evaluation completed!')
    elif args.mode == 'test':
        logging.info('Mode: Testing')

        # Load trained model from file
        model = load_model(args.model)

        # Load normalization stats from file
        normalization_stats = loadDictJson(stats_json)

        # Load user-defined or system-generated anomalies thresholds for testing
        thresholds_dict = loadDictJson(args.thresholds_file)

        # Create an empty dataframe to save fetched metrics
        metrics_cols = 34
        metrics = createEmptyMetricsArray(col_num=metrics_cols)

        # Keep last fetched record in order to create useful metrics.
        # If it is none make two requests for metrics
        last_record = None

        # Check influx connection. Create any necessary tables, if there are not exist
        push_anomalies_to_influx = True
        client = initializeConnection(args.influx_host, args.influx_port, args.influx_user, args.influx_pass)
        try:
            checkDatabase(client, args.influx_db)
            client.switch_database(args.influx_db)
        except Exception as e:
            push_anomalies_to_influx = False

            logging.error('Anomalies\' table does not exists or is not accessible from application')
            logging.error('Detected anomalies will be not pushed in InfluxDB')

        influx_queries = [
            "SELECT value FROM node_cpu_seconds_total WHERE mode='user' AND cpu='0' ORDER BY time DESC LIMIT 1;",
            "SELECT value FROM node_cpu_seconds_total WHERE mode='user' AND cpu='1' ORDER BY time DESC LIMIT 1;",
            "SELECT value FROM rx_cpu_time ORDER BY time LIMIT 1;",
            "SELECT value FROM tx_cpu_time ORDER BY time LIMIT 1;",
            "SELECT value FROM node_network_receive_bytes_total WHERE device='enp1s0' ORDER BY time DESC LIMIT 1;",
            "SELECT value FROM node_network_receive_bytes_total WHERE device='enp0s20u1' ORDER BY time DESC LIMIT 1;",
            "SELECT value FROM node_network_receive_bytes_total WHERE device='ppp0' ORDER BY time DESC LIMIT 1;",
            "SELECT value FROM node_network_transmit_bytes_total WHERE device='enp1s0' ORDER BY time DESC LIMIT 1;",
            "SELECT value FROM node_network_transmit_bytes_total WHERE device='enp0s20u1' ORDER BY time DESC LIMIT 1;",
            "SELECT value FROM node_network_transmit_bytes_total WHERE device='ppp0' ORDER BY time DESC LIMIT 1;",
            "SELECT value FROM ul_bitrate WHERE cellId='2' ORDER BY time DESC LIMIT 1;",
            "SELECT value FROM dl_bitrate WHERE cellId='2' ORDER BY time DESC LIMIT 1;",
            "SELECT value FROM node_memory_MemFree_bytes ORDER BY time DESC LIMIT 1;"
        ]

        influx_queries_measurements = [
            'node_cpu_seconds_total',
            'node_cpu_seconds_total',
            'rx_cpu_time',
            'tx_cpu_time',
            'node_network_receive_bytes_total',
            'node_network_receive_bytes_total',
            'node_network_receive_bytes_total',
            'node_network_transmit_bytes_total',
            'node_network_transmit_bytes_total',
            'node_network_transmit_bytes_total',
            'ul_bitrate',
            'dl_bitrate',
            'node_memory_MemFree_bytes'
        ]

        # Testing pipeline (execute every 10 seconds)
        while True:
            # Fetch metrics from influx
            curr_metrics, timestamp = getLastRecords(client, influx_queries, influx_queries_measurements)

            if last_record is None:
                last_record = curr_metrics
                time.sleep(refresh_time_interval)
                curr_metrics, timestamp = getLastRecords(client, influx_queries, influx_queries_measurements)

            # Add new line for fetched metrics in array
            metrics = np.vstack((metrics, np.zeros(shape=(1,metrics_cols))))
            metrics[-1, 0] = timestamp

            # Extract info from fetched metrics
            # First position in metrics array is timestamp
            # Positions 1-4, 20-21 are cpu preprocessed metrics
            # Positions 17-19 are memory preprocessed metrics
            # Positions 5-10, 11-16, 22-23, 24-25 are network received & transmitted metrics
            # Positions 26-27 are 5G uplink bytes_count and Bps. Positions 28-29 are 5G downlink bytes_count and Bps
            # Positions 30-31 are 5G RX cpu percentage & rate. Positions 32-33 5G TX cpu percentage & rate
            last_record['node_cpu_seconds_total'], metrics[-1, 1:5], metrics[-1, 20:22] = formatCpuSecondsTotal(curr_metrics['node_cpu_seconds_total'], last_record['node_cpu_seconds_total'])
            metrics[-1, 17:20] = formatMemoryFreeBytes(curr_metrics['node_memory_MemFree_bytes'], last_record['node_memory_MemFree_bytes'])
            metrics[-1, 5:11], metrics[-1, 22:24] = formatNetworkBytes(curr_metrics['node_network_receive_bytes_total'], last_record['node_network_receive_bytes_total'])
            metrics[-1, 11:17], metrics[-1, 24:26] = formatNetworkBytes(curr_metrics['node_network_transmit_bytes_total'], last_record['node_network_transmit_bytes_total'])
            metrics[-1, 26:28] = format5gNetworkBytes(curr_metrics['ul_bitrate'], last_record['ul_bitrate'])
            metrics[-1, 28:30] = format5gNetworkBytes(curr_metrics['dl_bitrate'], last_record['dl_bitrate'])
            metrics[-1, 30:32] = format5gCpuPercentage(curr_metrics['rx_cpu_time'], last_record['rx_cpu_time'])
            metrics[-1, 32:34] = format5gCpuPercentage(curr_metrics['tx_cpu_time'], last_record['tx_cpu_time'])

            # Set last record
            last_record = curr_metrics

            # Check sliding time window before appending new data. If it is full, remove oldest record
            if len(metrics) > time_window_threshold:
                metrics = metrics[-time_window_threshold:]

            # Convert to pandas df
            test_df = convertNumpyToPandas(metrics)

            # Normalize features
            for col in cols_to_normalize:
                test_df[col + '_normalized'] = normalizeFeature(test_df, col, normalization_stats[col + '_min'], normalization_stats[col + '_max'])

            # Reshape test dataset to fit model
            test_dataset = test_df[cols].to_numpy()
            X_test, y_test = split_sequences(test_dataset, n_steps)
            X_test = X_test.reshape((len(X_test), n_steps, n_features))

            # Predict
            y = model.predict(X_test, verbose=0)

            if (len(y_test) > 0) and (len(y) > 0):
                # Calculate total RMSE
                rmse_total = mean_squared_error(y_test, y, squared=False)

                # Calculate RMSE for each feature
                features_rmse = {}
                features_rmse['cpu'] = sqrt(mean_squared_error(y_test[:,0], y[:,0]))
                features_rmse['cpu_rx'] = sqrt(mean_squared_error(y_test[:,1], y[:,1]))
                features_rmse['cpu_tx'] = sqrt(mean_squared_error(y_test[:,2], y[:,2]))
                features_rmse['net_down'] = sqrt(mean_squared_error(y_test[:,3], y[:,3]))
                features_rmse['net_up'] = sqrt(mean_squared_error(y_test[:,4], y[:,4]))
                features_rmse['net_5g_down'] = sqrt(mean_squared_error(y_test[:,5], y[:,5]))
                features_rmse['net_5g_up'] = sqrt(mean_squared_error(y_test[:,6], y[:,6]))
                features_rmse['mem'] = sqrt(mean_squared_error(y_test[:,7], y[:,7]))

                # String with suggestions, based on threshold
                possible_causes = ''

                # Check if calculated RMSEs are over threshold
                is_anomaly = False
                for f in features_rmse.keys():
                    if features_rmse[f] > thresholds_dict[f+'_threshold']:
                        is_anomaly = True
                        possible_causes = possible_causes + str(f) + ' | '
                
                if rmse_total > thresholds_dict['overall_threshold']:
                    is_anomaly = True

                if possible_causes == '':
                    possible_causes = 'unknown cause'

                push_anomalies_to_influx = False
                if is_anomaly:
                    if push_anomalies_to_influx:
                        # Write anomalies to InfluxDB
                        anomaly_df = test_df.tail(1)[getAnomalyColumns()]
                        insertAnomalies(client, args.influx_measurement, anomaly_df.index, anomaly_df.to_numpy(), possible_causes)

            time.sleep(refresh_time_interval)
    else:
        logging.error('Unknown mode.')
        logging.error('Available modes: train/test')