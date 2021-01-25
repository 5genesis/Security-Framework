"""
InfluxDB Functions
"""

import logging
import time 

import numpy as np

from influxdb import InfluxDBClient

logging.getLogger('matplotlib').setLevel(level=logging.CRITICAL)
logging.getLogger("requests").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

def initializeConnection(influx_host, influx_port, influx_user, influx_pass):
    """
    Initialize connection with Influx DB.

    IMPORTANT: User must have rights to create a new database. 
    If user cannot create new database, detected anomalies will not be inserted in database.
    Database, can also created manually.

    param influx_host: Host's IP, where Influx is running.
    param influx_port: Host's port, where Influx service is listening.
    param influx_user: Username for Influx connection. 
    param influx_pass: User's password.

    return: Influx client object
    """

    client = InfluxDBClient(host=influx_host, port=influx_port, username=influx_user, password=influx_pass, ssl=False, verify_ssl=False)

    return client

def checkDatabase(client, db_name):
    """
    Checks if database, where metrics and detected anomalies will be stored, exists. 
    If database does not exists, it will be created. Otherwise return with no action.

    param client: Influx client. Client must be initialized and connection with Influx must be established.
    param db_name: Database name, where metrics and detected anomalies will be stored.

    return: None
    """

    for db in client.get_list_database():
        if db['name'] == db_name:
            logging.info('Database '+str(db_name)+' already exists')
            return

    logging.info('Creating database '+str(db_name))
    client.create_database(db_name)

def getLastRecords(client, queries, measurements):
    """
    Fetches last inserted record for each metric. Executes a list of quesries.
    Each executed query fetches the last inserted record for one measurement.

    param client: InfluxDB client object.
    param queries: List of influx queries, that will be executed.
    param mesaurements: list of measurements, that match the queries.

    return: data, timestamp
    data: A python dict containing all fetched data. Dict's key is measurement name and value the 'value' column of Influx DB.
    timestamp: A common timestamp for all measurements, to synchronize some of them that are inserted with a delay compared with the rest.
    """

    timestamp = time.time()

    data = {}
    idx = 0

    for q in queries:
        results = client.query(q).get_points()
        for r in results:
            if measurements[idx] not in data.keys():
                data[measurements[idx]] = []

            info = {}
            if measurements[idx] == 'node_cpu_seconds_total':
                info['mode'] = 'user'
                info['value'] = r['value']
                info['timestamp'] = timestamp
                if "cpu='0'" in q:
                    info['cpu'] = '0'
                elif "cpu='1'" in q:
                    info['cpu'] = '1'
            elif measurements[idx] in ['rx_cpu_time', 'tx_cpu_time']:
                info['value'] = r['value']
                info['timestamp'] = timestamp
            elif measurements[idx] in ['node_network_receive_bytes_total', 'node_network_transmit_bytes_total']:
                info['value'] = r['value']
                info['timestamp'] = timestamp
                if "device='enp1s0'" in q:
                    info['device'] = 'enp1s0'
                elif "device='enp0s20u1'" in q:
                    info['device'] = 'enp0s20u1'
                elif "device='ppp0'" in q:
                    info['device'] = 'ppp0'
            elif measurements[idx] in ['ul_bitrate', 'dl_bitrate']:
                info['value'] = r['value']
                info['timestamp'] = timestamp
                info['cellId'] = '2'
            elif measurements[idx] == 'node_memory_MemFree_bytes':
                info['value'] = r['value']
                info['timestamp'] = timestamp
            
            data[measurements[idx]].append(info)

        idx = idx + 1

    return data, timestamp


def insertAnomalies(client, influx_measurement, timestamps, anomaly_np, tag_msg):
    """
    Inserts the detcted anomalies in the defined measurement in Influx DB.

    param client: InfluxDB client object.
    param influx_measurement: Measurement, where the anomalies will be stored. It is defined as parameter when the main program starts.
    param timestamps: Timestamp of detected anomaly.
    param anomaly_np: A numpy array, that contains metrics' values for detected anomaly.
    param tag_msg: String, with a possible cause of detected anomalies. If there is no clear cause, its value will be 'unknown cause'. 

    return: None
    """

    anomalies_json = []

    for i in range(0, len(timestamps)):
        tags = {}
        tags['possible_cause'] = tag_msg

        fields = {}
        fields['cpu_usage'] = anomaly_np[i][0]
        fields['mem_usage'] = anomaly_np[i][1]
        fields['net_down_usage'] = anomaly_np[i][2]
        fields['net_up_usage'] = anomaly_np[i][3]
        fields['cpu_rx_usage'] = anomaly_np[i][4]
        fields['cpu_tx_usage'] = anomaly_np[i][5]
        fields['net_down_5g_usage'] = anomaly_np[i][6]
        fields['net_up_5g_usage'] = anomaly_np[i][7]

        a = {}
        a['measurement'] = influx_measurement
        a['tags'] = tags
        a['time'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        a['fields'] = fields

        anomalies_json.append(a)
    
    result = client.write_points(anomalies_json)

    if result:
        logging.info(str(len(timestamps)) + ' records pushed to InfluxDB')
    else:
        logging.error('An error occured. ' + str(len(timestamps)) + ' records ignored')