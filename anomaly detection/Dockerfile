FROM python:3.7.9-slim

# Set home directory
ENV AD_HOME=/home/5g_anomaly_detection

# Set path
ENV PATH=${PATH}:${AD_HOME}

# Create some needed paths
RUN mkdir -p /home/5g_anomaly_detection
RUN mkdir -p /home/5g_anomaly_detection/data
RUN mkdir -p /home/5g_anomaly_detection/model
RUN mkdir -p /home/5g_anomaly_detection/plots
RUN mkdir -p /home/5g_anomaly_detection/docs

# Copy local files to image
COPY *.py /home/5g_anomaly_detection/
COPY requirements.txt /home/5g_anomaly_detection/
COPY start_anomaly_detection.sh /home/5g_anomaly_detection/
COPY stop_anomaly_detection.sh /home/5g_anomaly_detection/
COPY data/* /home/5g_anomaly_detection/
COPY model/* /home/5g_anomaly_detection/model/
COPY plots/* /home/5g_anomaly_detection/plots/
COPY docs/* /home/5g_anomaly_detection/docs/

RUN chmod +x /home/5g_anomaly_detection/start_anomaly_detection.sh
RUN chmod +x /home/5g_anomaly_detection/stop_anomaly_detection.sh
RUN chmod -R 777 /home/5g_anomaly_detection

# Install all python dependencies
RUN pip install -r /home/5g_anomaly_detection/requirements.txt

############### EXPOSE ports ###############
# Port of Python server
EXPOSE 1234

############### Docker ENV params ###############
# Train model, if there is not a trained model saved
# Default: True
ENV TRAIN_MODEL True
# Evaluate model after training. Evaluation will use some cpu stress data, iperf stress data & training data. 
# Default: False
ENV EVAL_MODEL True

# Influx DB connection params. These must be provided by user.
# Default values in case user does not provide these params
# Host IP where Influx is running
ENV INFLUX_HOST localhost
# Port where Influx is running
ENV INFLUX_PORT 8086
# Influx user, that has access to data
ENV INFLUX_USER admin
# Influx user's password
ENV INFLUX_PASS admin
# Influx DB, where all data stored. This will be created, if not exists
# If user has no right to create the database, it must be created manually.
# Otherwise predicted anomalies will be not inserted in Influx
ENV INFLUX_DB metrics_db
# Influx measurement where detected anomalies will be inserted
ENV INFLUX_ANOMALIES_MEASUREMENT detected_anomalies

# Change directory to project root path
RUN cd /home/5g_anomaly_detection

# Run script, when container starts. Training model is optional
CMD ["/home/5g_anomaly_detection/start_anomaly_detection.sh"]
