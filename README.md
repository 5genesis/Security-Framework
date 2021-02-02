# Security-Framework

## Amarisoft exporter
Prometheus exporter for metrics provided by Amarisoft RAN socket API, written in `Go`.  
The exporter needs to be placed on a network accessible from Prometheus server and Amarisoft.
</br>

### Installation and Usage

Download and extract the latest executable for your architecture and edit the config.yml.  
Binaries availlable for `amd64`, `arm64` and `386` linux platforms.

**Example:**

Extract file:
> tar -xf amari-exporter-linux-amd64.tar.gz  
> cd  amari-exporter-linux-amd64
</br>

config.yml
```yaml
amari_url: ws://192.168.137.10:9001/
port: 3333
interval: 10s
```
</br>

Pass the config.yml as flag
> ./amarisoft-exporter-linux-amd64 -config config.yml
</br>

Amarisoft exporter is now running, you can verify metrics show up
> curl http://localhost:3333/metrics
</br>  

### Configure Prometheus server to scrape Amarisoft exporter
Add Amarisoft exporter to the list of prometheus targets  

prometheus.yml
```yaml
scrape_configs:
  - job_name: 'amari_exporter'
    scrape_interval: 10s
    static_configs:
      - targets: ["localhost:3333"]
```

### License
TBD
