package main

import (
	"fmt"
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	reg = prometheus.NewRegistry()

	dl_bitrate = promauto.With(reg).NewGaugeVec(prometheus.GaugeOpts{
		Name: "dl_bitrate",
		Help: "Downlink bitrate (bits/sec).",
	},
		[]string{"cellId"},
	)
	ul_bitrate = promauto.With(reg).NewGaugeVec(prometheus.GaugeOpts{
		Name: "ul_bitrate",
		Help: "Uplink bitrate (bits/sec).",
	},
		[]string{"cellId"},
	)
	dl_tx = promauto.With(reg).NewGaugeVec(prometheus.GaugeOpts{
		Name: "dl_tx",
		Help: "Downlink transmitted blocks number.",
	},
		[]string{"cellId"},
	)
	ul_tx = promauto.With(reg).NewGaugeVec(prometheus.GaugeOpts{
		Name: "ul_tx",
		Help: "Uplink transmitted blocks number.",
	},
		[]string{"cellId"},
	)
	rx_sample_rate = promauto.With(reg).NewGauge(prometheus.GaugeOpts{
		Name: "rx_sample_rate",
		Help: "CPU consumption for RX in Million samples per sec.",
	})
	tx_sample_rate = promauto.With(reg).NewGauge(prometheus.GaugeOpts{
		Name: "tx_sample_rate",
		Help: "CPU consumption for TX in Million samples per sec.",
	})
	rx_cpu_time = promauto.With(reg).NewGauge(prometheus.GaugeOpts{
		Name: "rx_cpu_time",
		Help: "CPU consumption for RX percentage.",
	})
	tx_cpu_time = promauto.With(reg).NewGauge(prometheus.GaugeOpts{
		Name: "tx_cpu_time",
		Help: "CPU consumption for TX percentage.",
	})
	rxtx_delay_min = promauto.With(reg).NewGauge(prometheus.GaugeOpts{
		Name: "rxtx_delay_min",
		Help: "TX-RX delay min.",
	})
	rxtx_delay_max = promauto.With(reg).NewGauge(prometheus.GaugeOpts{
		Name: "rxtx_delay_max",
		Help: "TX-RX delay max.",
	})
	rxtx_delay_avg = promauto.With(reg).NewGauge(prometheus.GaugeOpts{
		Name: "rxtx_delay_avg",
		Help: "TX-RX delay avg.",
	})
)

func main() {

	http.Handle("/metrics", promhttp.HandlerFor(
		reg,
		promhttp.HandlerOpts{
			// Opt into OpenMetrics to support exemplars.
			EnableOpenMetrics: true,
		}))

	go Exporter()
	fmt.Println("Listening on port :3333")
	panic(http.ListenAndServe(":3333", nil))
}
