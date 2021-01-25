#!/bin/bash

cd ${AD_HOME}

# Run python http server for docs in background.
echo "Starting docs python HTTP Server in port 1234"
python -m http.server 1234 -d docs/ &> /dev/null &

THRESHOLDS_FILENAME="data/thresholds.json"
THRESHOLDS_JSON_STRING="{"

# Check if threshold variables are not empty. If condition is true these values are set from user as ENV params
if [ ! -z ${CPU_TH} ]; then
		THRESHOLDS_JSON_STRING+="\"cpu_threshold\": ${CPU_TH}, "
fi
if [ ! -z ${MEM_TH} ]; then
		THRESHOLDS_JSON_STRING+="\"mem_threshold\": ${MEM_TH}, "
fi
if [ ! -z ${CPU_RX_TH} ]; then
		THRESHOLDS_JSON_STRING+="\"cpu_rx_threshold\": ${CPU_RX_TH}, "
fi
if [ ! -z ${CPU_TX_TH} ]; then
		THRESHOLDS_JSON_STRING+="\"cpu_tx_threshold\": ${CPU_TX_TH}, "
fi
if [ ! -z ${NET_UP_TH} ]; then
		THRESHOLDS_JSON_STRING+="\"net_up_threshold\": ${NET_UP_TH}, "
fi
if [ ! -z ${NET_DOWN_TH} ]; then
		THRESHOLDS_JSON_STRING+="\"net_down_threshold\": ${NET_DOWN_TH}, "
fi
if [ ! -z ${NET_5G_UP_TH} ]; then
		THRESHOLDS_JSON_STRING+="\"net_5g_up_threshold\": ${NET_5G_UP_TH}, "
fi
if [ ! -z ${NET_5G_DOWN_TH} ]; then
		THRESHOLDS_JSON_STRING+="\"net_5g_down_threshold\": ${NET_5G_DOWN_TH}, "
fi
if [ ! -z ${OVERALL_TH} ]; then
		THRESHOLDS_JSON_STRING+="\"overall_threshold\": ${OVERALL_TH}, "
fi

THRESHOLDS_JSON_STRING=${THRESHOLDS_JSON_STRING::-2}
THRESHOLDS_JSON_STRING+="}"

if [ $THRESHOLDS_JSON_STRING != "{}" ]; then
  echo $THRESHOLDS_JSON_STRING > $THRESHOLDS_FILENAME
fi

cat $THRESHOLDS_FILENAME

# If train is enabled, train model before using deploying it for testing
if [[ ${TRAIN_MODEL} == "True" || ${TRAIN_MODEL} == "true" ]]; then
	echo "Training Model"
	if [[ ${EVAL_MODEL} == "True" || ${EVAL_MODEL} == "true" ]]; then
		# In case evaluation is enabled, thresholds that are not set by user will be set from evaluation operation
		python find_anomalies.py \
			--mode train \
			--model model/5g_autoencoder.h5 \
			--evaluate true \
			--thresholds_file $THRESHOLDS_FILENAME
	else
		# In case evaluation is disbaled, thresholds that are not set by user will be set with default values
		python find_anomalies.py \
			--mode train \
			--model model/5g_autoencoder.h5 \
			--evaluate false \
			--thresholds_file $THRESHOLDS_FILENAME

		# Thresholds for prediction. Evaluation will suggest some thresholds if is enabled. Otherwise user can set custom as ENV param.
		# If EVAL_MODEL is False & user does not provide custom thresholds some default values will be used.

		# Threshold about prediction error in CPU usage in edge machine
		if [[ ${CPU_TH} == "" ]]; then
			echo "CPU_TH is not set. Using default value: 0.1"
			CPU_TH=0.1
		fi
		# Threshold about prediction error in memory usage in edge machine
		if [[ ${MEM_TH} == "" ]]; then
			echo "MEM_TH is not set. Using default value: 0.1"
			MEM_TH=0.1
		fi

		# Threshold about prediction error in RX CPU usage in RAN machine
		if [[ ${CPU_RX_TH} == "" ]]; then
			echo "CPU_RX_TH is not set. Using default value: 0.1"
			CPU_RX_TH=0.1
		fi
		# Threshold about prediction error in TX CPU usage in RAN machine
		if [[ ${CPU_TX_TH} == "" ]]; then
			echo "CPU_TX_TH is not set. Using default value: 0.1"
			CPU_TX_TH=0.1
		fi

		# Threshold about prediction error in network upload in edge machine
		if [[ ${NET_UP_TH} == "" ]]; then
			echo "NET_UP_TH is not set. Using default value: 0.1"
			NET_UP_TH=0.1
		fi
		# Threshold about prediction error in network download in edge machine
		if [[ ${NET_DOWN_TH} == "" ]]; then
			echo "NET_DOWN_TH is not set. Using default value: 0.1"
			NET_DOWN_TH=0.1
		fi

		# Threshold about prediction error in 5G network upload in RAN machine
		if [[ ${NET_5G_UP_TH} == "" ]]; then
			echo "NET_5G_UP_TH is not set. Using default value: 0.1"
			NET_5G_UP_TH=0.1
		fi
		# Threshold about prediction error in 5G network download in RAN machine
		if [[ ${NET_5G_DOWN_TH} == "" ]]; then
			echo "NET_5G_DOWN_TH is not set. Using default value: 0.1"
			NET_5G_DOWN_TH=0.1
		fi

		# Threshold about prediction error in all features used for testing. 
		# Useful for detecting combined attacks, where each feature's metric is not very abnormal
		if [[ ${OVERALL_TH} == "" ]]; then
			echo "OVERALL_TH is not set. Using default value: 0.1"
			OVERALL_TH=0.1
		fi

		echo "{cpu_threshold: ${CPU_TH}, mem_threshold: ${MEM_TH}, cpu_rx_threshold: ${CPU_RX_TH}, cpu_tx_threshold: ${CPU_TX_TH}, " \
			"net_up_threshold: ${NET_UP_TH}, net_down_threshold: ${NET_DOWN_TH}, net_5g_up_threshold: ${NET_5G_UP_TH}, " \
			"net_5g_down_threshold: ${NET_5G_DOWN_TH}, overall_threshold: ${OVERALL_TH}" > $THRESHOLDS_FILENAME
	fi
fi

cat $THRESHOLDS_FILENAME

echo "Starting trained model for predicting"
python find_anomalies.py \
	--mode test \
	--model model/5g_autoencoder.h5 \
	--thresholds_file $THRESHOLDS_FILENAME \
	--influx_host ${INFLUX_HOST} \
	--influx_port ${INFLUX_PORT} \
	--influx_user ${INFLUX_USER} \
	--influx_pass ${INFLUX_PASS} \
	--influx_db ${INFLUX_DB} \
	--influx_measurement ${INFLUX_ANOMALIES_MEASUREMENT}