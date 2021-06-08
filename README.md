# 5Genesis Security-Framework Release B


## Architecture

![alt text](https://github.com/5genesis/Security-Framework/blob/Release_B/anomaly%20detection/ui/static/images/SecFrameworkArch.png?raw=true)

There are three main components Amarisoft RAN, Edge Cloud node and Core DC where Security Framework tools are deployed. Amarisoft RAN provides an API for collecting 5G metrics. In Edge node a Prometheus server and the two exporters are running. In Core DC an Influx DB is deployed. Also, the anomaly detection software and a provided UI through Grafana are deployed. Amarisoft Exporter collects 5G metrics from Amarisoft RAN through a socket. Node Exporter collects metrics from the Edge Node and Prometheus server scrapes both exporters. Prometheus publishes collected metrics to Influx DB every 15 seconds. Anomaly Detection model fetches ingested data from Influx DB and decides if a record is an anomaly compared to normal records or not. Detected anomalies are saved to Influx DB. Grafana UI depicts metrics gathered from Edge Node and Amarisoft, and also keeps a table with all detected anomalies. For each anomaly there is the feature to show more info about collected and monitored metrics.

## Edge Cloud

### Requirements

On Edge node Prometheus server and the two exporters need to be installed.

### Node Exporter

Node exporter collects OS metrics from the underlying Edge Cloud machine.

Install Node exporter on Edge node:
https://prometheus.io/docs/guides/node-exporter/

### Amarisoft Exporter

Prometheus exporter for metrics provided by Amarisoft RAN socket API, written in `Go`. The exporter needs to be placed on a network accessible from Prometheus server and Amarisoft.

#### Installation and Usage

[Download] and extract the latest executable for your architecture and edit the config.yml.
Binaries availlable for amd64, arm64 and 386 linux platforms.

##### Example:

- Extract binary:
```
tar -xf amari-exporter-linux-amd64.tar.gz
cd amari-exporter-linux-amd64
```
- Specify the port exporter is running on Edge node, the interval to query Amarisoft API and Amarisoft url
config.yml

Specify the port exporter is running on Edge node, the interval to query Amarisoft API and Amarisoft url
config.yml
```
amari_url: ws://192.168.137.10:9001/
port: 3333
interval: 10s
```

- Pass the config.yml as flag to binary
```
./amarisoft-exporter-linux-amd64 -config config.yml
```
- Amarisoft exporter is now running, you can verify metrics show up
```
curl http://localhost:3333/metrics
``` 

### Prometheus server

Prometheus server scrapes both exporters (targets), performs filtering on collected metrics and publishes to remote Influx DB.

- Install Prometheus server either as [standalone] or using [Docker image]
- Configure targets to scrape, remote write to Influx DB and metrics filtering using a config.yml:
```
# Remote write configuration for Influx
remote_write:
  - url: "http://{influx_host}:{port}/api/v1/prom/write?db={db}&u={user}&p={pass}"
scrape_configs:
  - job_name: ‘amari exporter’
    scrape_interval: 15s
    static_configs:
      - targets: ["{url}:{port}"]
  - job_name: 'node_exporter_dellEdge'
    scrape_interval: 15s
    static_configs:
      - targets: ["{url}:{port}"]
    metric_relabel_configs:
      - source_labels: [__name__]
        regex: 'node_cpu_seconds_total|node_memory_Buffers_bytes|node_memory_Cached_bytes|node_memory_Mem.+|node_memory_Swap.+|node_network_receive_bytes_total|node_network_transmit_bytes_total|node_filesyste$
        action: keep
```

##### Example:

prometheus.yml
```
# Remote write configuration for Influx
remote_write:
  - url: "http://10.10.x.x:8086/api/v1/prom/write?db=prometheus_db&u=user&p=passwd"
scrape_configs:
  - job_name: 'amari_exporter'
    scrape_interval: 15s
    static_configs:
      - targets: ["localhost:3333"]
  - job_name: 'node_exporter_dellEdge'
    scrape_interval: 15s
    static_configs:
      - targets: ["localhost:9100"]
    metric_relabel_configs:
      - source_labels: [__name__]
        regex: 'node_cpu_seconds_total|node_memory_Buffers_bytes|node_memory_Cached_bytes|node_memory_Mem.+|node_memory_Swap.+|node_network_receive_bytes_total|node_network_transmit_bytes_total|node_filesyste$
        action: keep
```

## Autoencoder

Autoencoder is the core component of Security Framework, encapsulates the detection logic and produces the results visualized on Grafana dashboard.

Below are the stages that autoencoder functions:

### Preprocessing

Training data have been collected from both Node Exporter and Amari Exporter. Because of these two different systems, there is a small difference in timestamps for these metrics. For this reason the training dataset is resampled every 15 seconds, to synchronize records from these two exporters. For training, the following features have been selected: CPU percentage rate, RAM percentage rate, RX/TX CPU percentage rate, rate of transmitted/received bytes and rate of bytes downloaded/uploaded from 5G interface. These features are normalized using Min-Max Normalization. The min & max values that will be used for normalization will be saved in a JSON file. After resampling and normalize the dataset is split in sequences, with four steps per sequence, where the first three records will be used to predict the fourth.

### Training

For training the architecture of an Autoencoder has been used. Model consisted of 9 Bidirectional LSTM layers and one Dense layer at the end.

![alt text](https://github.com/5genesis/Security-Framework/blob/Release_B/anomaly%20detection/ui/static/images/training1.png?raw=true)

As activation function ReLu is used. SGD with Nesterov momentum has been selected as optimizer, with learning rate equal to 0.01. As validation dataset 10% of the trainset has been used. The model is trained for 50 epochs.

![alt text](https://github.com/5genesis/Security-Framework/blob/Release_B/anomaly%20detection/ui/static/images/training.png?raw=true)

### Evaluation

If evaluation is enabled, trained model will be loaded and used for testing in three different datasets. The first dataset has been collected during a CPU stress test attack. The second dataset has been collected during an iperf stress test attack. The third dataset is the training set. All these datasets will be normalized with saved normalization values during training. Using these datasets, the RMSE between the actual and predicted values will be calculated. Network features’ thresholds, will not be affected from CPU stress dataset and CPU features’ thresholds will not be affected from iperf stress test dataset. The 99th percentile of each feature’s RMSE will be considered as threshold from anomaly detection algorithm. The 99th percentile has been selected compared with the max value in order to avoid outliers, that may exist in these datasets. User-defined thresholds will not be overridden from calculated RMSEs.

### Testing

Trained model will be used in order to predict the next features’ values in the time series. Because it is trained in normal traffic it will predict these values considering that the incoming traffic is normal. If the RMSE between the predicted and actual value is above a set threshold this record will be considered as anomaly. Algorithm fetches the last received from an Influx DB every 15 seconds. For predicting the next value, a sliding window with size equal to 30, keeping only last records is used. Fetched records are been preprocessed and normalized before used by the model for predicting. After prediction the RMSEs between the actual and predicted values are calculated and compared with set thresholds, to identify if the record considers an anomaly or not. A possible cause of the anomaly is also saved in the database with the detected anomalies or an ‘unknown cause’ will be set if the cause cannot be identified. A proposed thresholds JSON file for testing is the follow:
```
{"cpu_threshold": 0.05, "mem_threshold": 0.1, "cpu_tx_threshold": 0.1, 
"cpu_rx_threshold": 0.1, "net_up_threshold": 0.501, 
"net_down_threshold": 0.501, "net_5g_up_threshold": 0.501, 
"net_5g_down_threshold": 0.501, "overall_threshold": 0.15}
```

## Core DC

### Prerequisites

On the Core DC machine we need Docker, Python3.7, Grafana, Influx DB installed

#### Docker

- Install Docker for your operating system ([https://docs.docker.com/get-docker/])
- Build Docker container from Dockerfile (from directory, where Dockerfile is located):
```
docker build –t 5g_anomaly_detection –f Dockerfile .
```

#### Python

- Tested with Python 3.7.0 and pip 10.0.1
- Required Python packages: pandas, numpy, scikit-learn, jsonschema, matplotlib, keras (2.3.1), tensorflow (2.2.1), tensorflow-cpu (2.2.1), h5py, influxdb
- All required packages with tested versions can be installed from requirements.txt file using the command (from directory, where requirements.txt is located):
```
pip install –r requirements.txt
```

#### Grafana

Grafana can be installed either locally ([https://grafana.com/docs/grafana/latest/installation/]) or within a Docker container ([https://hub.docker.com/r/grafana/grafana]).
After installation, two data sources must be set: a Prometheus data source and an InfluxDB data source. After that the created dashboard can be imported from a json file (ui/threat-detection-ui.json).

#### Influx DB

- Install Influx DB [https://docs.influxdata.com/influxdb/v2.0/get-started/]
- Create a database for prometheus metrics to be stored as measurements

In addition, there are certain columns in some measurements, which require specific values:
- node_cpu_seconds_total:
    - cpu: must has values 0 and 1. These two are used for collecting and aggregating metrics.
- node_network_receive_bytes_total/node_network_transmit_bytes_total:
    - device: enp1s0, enp0s20u1 and ppp0 are used for collecting metrics and aggregating received and transmitted bytes and their rate.
- ul_bitrate/dl_bitrate:
    - cellId: There are two values for cellId: 1 and 2. cellId 1 is used for LTE connections and cellId 2 is used for 5G connections. The algorithms collects metrics only from cellId 2, which are 5G related.

![alt text](https://github.com/5genesis/Security-Framework/blob/Release_B/anomaly%20detection/ui/static/images/Influx.png?raw=true)

### Execution

#### Docker

- Run the docker container (in the background) with the following command:
    - docker run –d –env ${ENV_PARAM_NAME_1}=${ENV_PARAM_VALUE_1} –env … -p ${HOST_PORT}:1234 5g_anomaly_detection
- Docker ENV params:
    - TRAIN_MODEL: Train Model before starting it for predicting in real time. (Default: True)
    - EVAL_MODEL: Evaluate Model in order to suggest some missing thresholds, that are not user defined. If all thresholds are set as ENV params this function can be avoid, because it will not override user-defined values. If EVAL_MODEL is set to False and user has not set some thresholds they will be set with a default value equal to 0.1. (Default: True)
    - CPU_TH: RMSE anomaly threshold for predicted CPU percentage rate in user mode. (Default: 0.1)
    - MEM_TH: RMSE anomaly threshold for predicted RAM percentage rate. (Default: 0.1)
    - CPU_RX_TH: RMSE anomaly threshold for predicted RX CPU percentage rate. (Default: 0.1)
    - CPU_TX_TH: RMSE anomaly threshold for predicted TX CPU percentage rate. (Default: 0.1)
    - NET_UP_TH: RMSE anomaly threshold for predicted bytes transmitted rate for selected interfaces. (Default: 0.1)
    - NET_DOWN_TH: RMSE anomaly threshold for predicted bytes received rate for selected interfaces. (Default: 0.1)
    - NET_5G_UP_TH: RMSE anomaly threshold for predicted bytes transmitted rate for 5G cell. (Default: 0.1)
    - NET_5G_DOWN_TH: RMSE anomaly threshold for predicted bytes received rate for 5G cell. (Default: 0.1)
    - OVERALL_TH: RMSE anomaly threshold for all predicted features aggregated. (Default: 0.1)
    - INFLUX_HOST: IP of machine where InfluxDB is running. (Default: localhost)
    - INFLUX_PORT: Port, where InfluxDB is listening. (Default: 8086)
    - INFLUX_USER: Username for connecting in InfluxDB. (Default: admin)
    - INFLUX_PASS: Password for connecting in InfluxDB. (Default: admin)
    - INFLUX_DB: Database, where all collected metrics and detected anomalies are stored. (Default: metrics_db)
    - INFLUX_ANOMALIES_MEASUREMENT: Measurement in ${INFLUX_DB} where all detected anomalies will be saved. (Default: detected_anomalies)
- Docker ports:
    - Port 1234: A simple python http server is running in order to provide docs for reading

If there is no change in training process or in the training data, TRAIN_MODEL param can be set to False, as the root folder contains a trained and evaluated model in the given data

#### Python (optional)

If the algorithm is executed from Python command line in testing mode, the file thresholds.json in data folder must exists. If there is no such file in the data folder, it must be created manually. Its structure is the following:
```
{"cpu_threshold": 0.1, "mem_threshold": 0.1, ...}
```

The required keys for thresholds.json file are the following: cpu_threshold, mem_threshold, cpu_tx_threshold, cpu_rx_threshold, net_up_threshold, net_down_threshold, net_5g_up_threshold, net_5g_down_threshold, overall_threshold.

- Training Mode (from root folder, where find_anomalies.py is located):

```
python find_anomalies.py \  
            --mode train \
            --model model/5g_autoencoder.h5 \
            --evaluate false \
            --thresholds_file data/thresholds.json
```

- Training Mode with evaluation (from root folder, where find_anomalies.py is located):
```
python find_anomalies.py \
            --mode train \
            --model model/5g_autoencoder.h5 \
            --evaluate true \
            --thresholds_file data/thresholds.json
```

- Testing Mode (from root folder, where find_anomalies.py is located). The following parameters must be set either as variables in shell or directly in the following command to execute the algorithm in test mode:
    - ${INFLUX_HOST}: IP of machine where InfluxDB is running.
    - ${INFLUX_PORT}: Port, where InfluxDB is listening.
    - ${INFLUX_USER}: Username for connecting in InfluxDB.
    - ${INFLUX_PASS}: Password for connecting in InfluxDB.
    - ${INFLUX_DB}: Database, where all collected metrics and detected anomalies are stored.
    - ${INFLUX_ANOMALIES_MEASUREMENT}: Measurement in ${INFLUX_DB} where all detected anomalies will be saved.
```
python find_anomalies.py \
    --mode test \
    --model model/5g_autoencoder.h5 \
    --thresholds_file data/thresholds.json \
    --influx_host ${INFLUX_HOST} \
    --influx_port ${INFLUX_PORT} \
    --influx_user ${INFLUX_USER} \
    --influx_pass ${INFLUX_PASS} \
    --influx_db ${INFLUX_DB} \
    --influx_measurement ${INFLUX_ANOMALIES_MEASUREMENT}
```

#### Grafana

After configuring data sources and importing the custom dashboard we get the following image:

![alt text](https://github.com/5genesis/Security-Framework/blob/Release_B/anomaly%20detection/ui/static/images/Grafana.png?raw=true)

[Download]: https://github.com/5genesis/Security-Framework/releases
[standalone]: https://prometheus.io/docs/prometheus/latest/getting_started/
[Docker image]: https://hub.docker.com/r/prom/prometheus/
[https://docs.docker.com/get-docker/]: https://docs.docker.com/get-docker/
[https://grafana.com/docs/grafana/latest/installation/]: https://grafana.com/docs/grafana/latest/installation/
[https://hub.docker.com/r/grafana/grafana]: https://hub.docker.com/r/grafana/grafana
[https://docs.influxdata.com/influxdb/v2.0/get-started/]: https://docs.influxdata.com/influxdb/v2.0/get-started/