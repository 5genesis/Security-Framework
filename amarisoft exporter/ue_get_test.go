package main

import (
	"io/ioutil"
	"testing"
)

func TestUE_getMessage(t *testing.T) {

	// Test message "stats" parsing
	// Mock socket response
	b, err := ioutil.ReadFile("ue_stats.json")
	if err != nil {
		println(err)
		return
	}

	reply := process(b)

	//handle each message accordingly
	messageType := reply["message"].(string)
	println("Received message: " + messageType)

	if messageType == "ue_get" {
		ueList := reply["ue_list"].([]interface{})
		// ueList iteration
		for _, value := range ueList {
			// cell values iteration
			ue := value.(map[string]interface{})
			var isRanMetric bool
			for k, v := range ue {
				if k == "ran_ue_id" {
					isRanMetric = true
				} else if k == "cells" {
					metrics := v.([]interface{})
					metric := metrics[0].(map[string]interface{})
					//var cell int8
					for name, vl := range metric {
						switch name {
						case "cell_id":
							println(vl.(float64))
						case "dl_bitrate":
							println(vl.(float64))
							println(isRanMetric)
						case "ul_bitrate":
							println(vl.(float64))
						case "dl_tx":
							println(vl.(float64))
						case "ul_tx":
							println(vl.(float64))
						}
					}
				}

			}
		}
	}

	//println(string(b))
	/*
		name := "Gladys"
		want := regexp.MustCompile(`\b` + name + `\b`)
		msg, err := Hello("Gladys")
		if !want.MatchString(msg) || err != nil {
			t.Fatalf(`Hello("Gladys") = %q, %v, want match for %#q, nil`, msg, err, want)
		}
	*/
}
