"""
Format retrieved metrics from InfluxDB
"""

import pandas as pd
import logging

logging.basicConfig(
    filename='5g_anomaly_detection.log',
    format='%(asctime)s [%(levelname)-8s] %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.getLogger('matplotlib').setLevel(level=logging.CRITICAL)
logging.getLogger("requests").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

def formatCpuSecondsTotal(json_array, last_record):
    """
    Receives CPU seconds total. Filter CPU seconds, keeping only 'user' mode. For both of the 2
    different CPUs calculate percentage usage using the time difference from the last_record param
    and the difference of seconds in 'user' mode. Also, based on last_record's percentage calculate
    the rate of the percentage for each CPU separately, and aggregated. Finally, return json_array 
    in order to be used as last_record for the next retrieved metrics.

    param json_array: JSON Array with total cpu seconds in node.
    param last_record: previous retrieved record from InfluxDB.

    return: json_array, features_core, features
    json_array: input param
    features_core: formatted percentage & rate for mode user for both cpu 0 and 1
    features: aggregated formatted percentage & rate for mode user
    """

    cpu_user_0_curr = [x for x in json_array if x['cpu']=='0' and x['mode']=='user'][0]
    cpu_user_1_curr = [x for x in json_array if x['cpu']=='1' and x['mode']=='user'][0]

    cpu_user_0_prev = [x for x in last_record if x['cpu']=='0' and x['mode']=='user'][0]
    cpu_user_1_prev = [x for x in last_record if x['cpu']=='1' and x['mode']=='user'][0]

    time_dif = cpu_user_0_curr['timestamp'] - cpu_user_0_prev['timestamp'] 

    cpu_user_0_seconds = cpu_user_0_curr['value'] - cpu_user_0_prev['value'] 
    cpu_user_1_seconds = cpu_user_1_curr['value'] - cpu_user_1_prev['value'] 

    cpu_user_0_percentage = (cpu_user_0_seconds / time_dif) * 100.0
    cpu_user_1_percentage = (cpu_user_1_seconds / time_dif) * 100.0
    
    if 'percentage' in cpu_user_0_prev.keys():
        cpu_user_0_percentage_prev = cpu_user_0_prev['percentage']
        cpu_user_1_percentage_prev = cpu_user_1_prev['percentage']
    else:
        cpu_user_0_percentage_prev = 0.0
        cpu_user_1_percentage_prev = 0.0
    
    cpu_user_0_rate = cpu_user_0_percentage - cpu_user_0_percentage_prev
    cpu_user_1_rate = cpu_user_1_percentage - cpu_user_1_percentage_prev

    cpu_user_percentage = (cpu_user_0_percentage + cpu_user_1_percentage) / 2
    cpu_user_rate = (cpu_user_0_rate + cpu_user_1_rate) / 2

    for j in json_array:
        if (j['cpu'] == '0') and (j['mode'] == 'user'):
            j['percentage'] = cpu_user_0_percentage
        elif (j['cpu'] == '1') and (j['mode'] == 'user'):
            j['percentage'] = cpu_user_1_percentage
    
    # Features array contains values that will be saved in metrics array
    features_core = [cpu_user_0_percentage, cpu_user_0_rate, cpu_user_1_percentage, cpu_user_1_rate]
    features = [cpu_user_percentage, cpu_user_rate]

    # Add percentage for each cpu in fetched metrics, and return it as last_record
    return json_array, features_core, features

def format5gCpuPercentage(json_array, last_record):
    """
    Receives percentage usage of RX/TX CPU. Calculate the rate of usage 
    between json_array and last_record.

    param json_array: JSON Array with percentage of RX/TX CPU.
    param last_record: previous retrieved record from InfluxDB.

    return: percentage of RX/TX CPU and rate of usage.
    """

    time_dif = json_array[0]['timestamp'] - last_record[0]['timestamp'] 

    cpu_curr = json_array[0]
    cpu_prev = last_record[0]

    cpu_percentage = json_array[0]['value']
    cpu_percentage_prev = last_record[0]['value']
        
    cpu_rate = cpu_percentage - cpu_percentage_prev

    features = [cpu_percentage, cpu_rate]

    return features

def formatMemoryFreeBytes(json_array, last_record):
    """
    Receives free bytes of available RAM. Total size of RAM is 8228077568.0 bytes.
    Substracting from total size calculate the the used RAM, percentage of RAM used.
    Using last_record calculate the rate of RAM usage.

    param json_array: JSON Array with number of free RAM bytes.
    param last_record: previous retrieved record from InfluxDB.

    return: Used RAM, percentage of used RAM and rate of RAM usage.
    """

    # Total RAM size. Having free RAM, we can calculate used RAM
    total_ram_bytes = 8228077568.0

    time_dif = json_array[0]['timestamp'] - last_record[0]['timestamp'] 

    mem_used = total_ram_bytes - json_array[0]['value']
    mem_used_prev = total_ram_bytes - last_record[0]['value']

    mem_used_percentage = (mem_used / total_ram_bytes) * 100.0

    if mem_used_prev == 0:
        mem_rate = mem_used_percentage
    else:
        mem_rate = ((mem_used - mem_used_prev) / mem_used_prev) * 100.0

    features = [mem_used, mem_used_percentage, mem_rate]

    return features

def formatNetworkBytes(json_array, last_record):
    """
    Receives bytes of all network interfaces. Filter them, and keep 'enp1s0', 'enp0s20u1' and 'ppp0'
    interfaces. It calculates using last_record bytes and bytes rate for each interface separately. 
    Also it calculates the total bytes and total rate aggregated from all three interfaces.

    param json_array: JSON Array with number of bytes in each interface.
    param last_record: previous retrieved record from InfluxDB.

    return: features_interfaces, features.
    features_interfaces: number of bytes and rate of bytes for each interface separately.
    features: Total number of bytes and aggregated rate of bytes from the three used interfaces.
    """

    enp1s0_dict = [x for x in json_array if x['device']=='enp1s0'][0]
    enp0s20u1_dict = [x for x in json_array if x['device']=='enp0s20u1'][0]
    ppp0_dict = [x for x in json_array if x['device']=='ppp0'][0]

    enp1s0_dict_prev = [x for x in last_record if x['device']=='enp1s0'][0]
    enp0s20u1_dict_prev = [x for x in last_record if x['device']=='enp0s20u1'][0]
    ppp0_dict_prev = [x for x in last_record if x['device']=='ppp0'][0]

    time_dif = enp1s0_dict['timestamp'] - enp1s0_dict_prev['timestamp'] 

    enp1s0_bytes = enp1s0_dict['value']
    enp1s0_bytes_prev = enp1s0_dict_prev['value']
    if enp1s0_bytes_prev == 0:
        enp1s0_bytes_rate = 100.0
        #enp1s0_bytes_rate = enp1s0_bytes / time_dif
    else:
        enp1s0_bytes_rate = ((enp1s0_bytes - enp1s0_bytes_prev) / enp1s0_bytes_prev) * 100.0
        #enp1s0_bytes_rate = (enp1s0_bytes - enp1s0_bytes_prev) / time_dif

    enp0s20u1_bytes = enp0s20u1_dict['value']
    enp0s20u1_bytes_prev = enp0s20u1_dict_prev['value']
    if enp0s20u1_bytes_prev == 0:
        enp0s20u1_bytes_rate = 100.0
        #enp0s20u1_bytes_rate = enp0s20u1_bytes / time_dif
    else:
        enp0s20u1_bytes_rate = ((enp0s20u1_bytes - enp0s20u1_bytes_prev) / enp0s20u1_bytes_prev) * 100.0
        #enp0s20u1_bytes_rate = (enp0s20u1_bytes - enp0s20u1_bytes_prev) / time_dif

    ppp0_bytes = ppp0_dict['value']
    ppp0_bytes_prev = ppp0_dict_prev['value']
    if ppp0_bytes_prev == 0:
        ppp0_bytes_rate = 100.0
        #ppp0_bytes_rate = ppp0_bytes / time_dif
    else:
        ppp0_bytes_rate = ((ppp0_bytes - ppp0_bytes_prev) / ppp0_bytes_prev) * 100.0
        #ppp0_bytes_rate = (ppp0_bytes - ppp0_bytes_prev) / time_dif

    enp1s0_bytes = enp1s0_bytes - enp1s0_bytes_prev
    enp0s20u1_bytes = enp0s20u1_bytes - enp0s20u1_bytes_prev
    ppp0_bytes = ppp0_bytes - ppp0_bytes_prev

    bytes_total = enp1s0_bytes + enp0s20u1_bytes + ppp0_bytes
    #bytes_rate = (enp1s0_bytes_rate + enp0s20u1_bytes_rate + ppp0_bytes_rate) / 3
    bytes_rate = enp1s0_bytes_rate + enp0s20u1_bytes_rate + ppp0_bytes_rate

    features_interfaces = [enp1s0_bytes, enp1s0_bytes_rate, enp0s20u1_bytes, enp0s20u1_bytes_rate, ppp0_bytes, ppp0_bytes_rate]
    features = [bytes_total, bytes_rate]

    return features_interfaces, features

def format5gNetworkBytes(json_array, last_record):
    """
    Receives bits of 5G interface.

    param json_array: JSON Array with number of bites for 5G interface.
    param last_record: previous retrieved record from InfluxDB.

    return: Total number of bytes and rate of bytes for 5G interface.
    """

    time_dif = json_array[0]['timestamp'] - last_record[0]['timestamp'] 

    bits_rate_per_second = json_array[0]['value']
    bytes_rate_per_second = json_array[0]['value']  / 8

    bits_rate_per_second_prev = last_record[0]['value']
    bytes_rate_per_second_prev = last_record[0]['value']  / 8

    bits_count = bits_rate_per_second * time_dif
    bytes_count = bytes_rate_per_second * time_dif

    bits_count_prev = bits_rate_per_second * time_dif
    bytes_count_prev = bytes_rate_per_second * time_dif

    if bytes_count_prev == 0:
        bytes_rate = 0.0
        bits_rate = 0.0
    else:
        bytes_rate = ((bytes_count - bytes_count_prev) / bytes_count_prev) * 100.0
        bits_rate = ((bits_count - bits_count_prev) / bits_count_prev) * 100.0

    features = [bytes_count, bytes_rate]

    return features
