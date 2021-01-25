package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"time"

	"github.com/gorilla/websocket"
)

// catchSig cleans up our websocket conenction if we kill the program
// with a ctrl-c
func catchSig(ch chan os.Signal, c *websocket.Conn) {
	// block on waiting for a signal
	<-ch
	err := c.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
	if err != nil {
		fmt.Println("write close:", err)
	}
	return
}

// Exporter gathers metrics from Amarisoft socket api
func Exporter() {

	// connect the os signal to our channel
	// interrupt := make(chan os.Signal, 1)
	// signal.Notify(interrupt, os.Interrupt)

	// connect the os signal to our channel
	//interrupt := make(chan os.Signal, 1)
	//signal.Notify(interrupt, os.Interrupt)

	// use the ws:// Scheme to connect to the websocket
	u := "ws://192.168.137.10:9001/"
	println("connecting to %s", u)

	c, _, err := websocket.DefaultDialer.Dial(u, http.Header{"origin": []string{"Test"}})
	if err != nil {
		log.Fatal("dial:", err)
	}
	defer c.Close()

	// dispatch our signal catcher
	// go catchSig(interrupt, c)

	//call Amarisoft Socket Api every 10 sec
	for {
		callAmariSocket(c)
		time.Sleep(10 * time.Second)
	}

	// in case Close fails we return the error
	println("Closing socket ...")
	c.Close()
}
func callAmariSocket(c *websocket.Conn) {

	var pending int = 0
	// Send message {"message":"stats","samples":true} to Amarisoft socket, similar to 't spl' command
	statsMessageId := "stats_msg"
	statsData := fmt.Sprintf("{\"message\": \"stats\",\"message_id\": \"%s\",\"rf\": true}", statsMessageId)

	err := c.WriteMessage(websocket.TextMessage, []byte(statsData))
	pending++
	if err != nil {
		log.Println("failed to write message:", err)
		return
	}

	// Send message {"message":"ue_get","stats":true} to Amarisoft socket, similar to 't ue' command
	ueMessageId := "ue_get_msg"
	/*		ue_getData := fmt.Sprintf("{\"message\": \"ue_get\",\"message_id\": \"%s\",\"stats\": true}", ueMessageId)
			err = c.WriteMessage(websocket.TextMessage, []byte(ue_getData))
			if err != nil {
				log.Println("failed to write message:", err)
				return
			}
	*/
	// read replies from server to find response
	var reply map[string]interface{}
	for pending > 0 {

		_, message, err := c.ReadMessage()
		if err != nil {
			log.Println("failed to read:", err)
			return
		}

		// unmarshal response into a map
		reply = process(message)
		if reply == nil {
			return
		}

		// process response according to message type ("stats" or "ue_get")
		if "stats" == reply["message"].(string) {
			processStatsMsg(statsMessageId, &reply)
			pending--
		} else if "ue_get" == reply["message"].(string) {
			processUEGetMsg(ueMessageId, &reply)
			pending--
		}

	}
}
func processStatsMsg(messageId string, reply *map[string]interface{}) {
	println("stats message received")
	if messageId == (*reply)["message_id"].(string) {
		//out, _ := json.MarshalIndent(reply, " ", "	")
		//println(string(out))
		if (*reply)["cells"] != nil {
			cells := (*reply)["cells"].(map[string]interface{})
			// iterate through cells
			for cellId, value := range cells {
				// cell values iteration
				cell := value.(map[string]interface{})
				//println(v.(float64))
				for k, v := range cell {
					switch k {
					case "dl_bitrate":
						dl_bitrate.WithLabelValues(cellId).Set(math.Round(100*v.(float64)) / 100)
					case "ul_bitrate":
						ul_bitrate.WithLabelValues(cellId).Set(math.Round(100*v.(float64)) / 100)
					case "dl_txok":
						dl_tx.WithLabelValues(cellId).Set(math.Round(100*v.(float64)) / 100)
					case "ul_rxok":
						ul_tx.WithLabelValues(cellId).Set(math.Round(100*v.(float64)) / 100)
					}
				}
			}
		} else {
			println("received empty cells reply")
		}
		if (*reply)["rf"] != nil {
			rf := (*reply)["rf"].(map[string]interface{})
			// cells iteration
			for field, v := range rf {
				switch field {
				case "rx_sample_rate":
					rx_sample_rate.Set(math.Round(10000*v.(float64)) / 10000)
				case "tx_sample_rate":
					tx_sample_rate.Set(math.Round(10000*v.(float64)) / 10000)
				case "rx_cpu_time":
					rx_cpu_time.Set(math.Round(100*v.(float64)) / 100)
				case "tx_cpu_time":
					tx_cpu_time.Set(math.Round(100*v.(float64)) / 100)
				case "rxtx_delay_min":
					rxtx_delay_min.Set(math.Round(100*v.(float64)) / 100)
				case "rxtx_delay_max":
					rxtx_delay_max.Set(math.Round(100*v.(float64)) / 100)
				case "rxtx_delay_avg":
					rxtx_delay_avg.Set(math.Round(100*v.(float64)) / 100)
				}
			}
		} else {
			println("received empty rf reply")
		}
	}
}
func processUEGetMsg(messageId string, reply *map[string]interface{}) {
	println("ue_get message received")
	if messageId == (*reply)["message_id"].(string) {
		ueList := (*reply)["ue_list"].([]interface{})
		// ueList iteration
		for _, value := range ueList {
			// cell values iteration
			ue := value.(map[string]interface{})
			var isRanMetric string
			for k, v := range ue {
				if k == "ran_ue_id" {
					//ueId = fmt.Sprintf("%.0f", v.(float64))
					isRanMetric = "true"
				} else if k == "cells" {
					metrics := v.([]interface{})
					metric := metrics[0].(map[string]interface{})
					var cellId string
					for name, vl := range metric {
						switch name {
						case "cell_id":
							cellId = fmt.Sprintf("%.0f", vl.(float64))
						case "dl_bitrate":
							dl_bitrate.WithLabelValues(isRanMetric, cellId).Set(vl.(float64))
						case "ul_bitrate":
							ul_bitrate.WithLabelValues(isRanMetric, cellId).Set(vl.(float64))
						case "dl_tx":
							dl_tx.WithLabelValues(isRanMetric, cellId).Set(vl.(float64))
						case "ul_tx":
							ul_tx.WithLabelValues(isRanMetric, cellId).Set(vl.(float64))
						}
					}
				}

			}
		}
	}
}
func process(response []byte) map[string]interface{} {

	/*** unmarshal json to a map ***/
	var result map[string]interface{}

	err := json.Unmarshal([]byte(response), &result)
	if err != nil {
		println("failed to process reply", err)
		return nil
	}

	return result
}
