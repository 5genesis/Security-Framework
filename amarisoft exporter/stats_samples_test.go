package main

import (
	"io/ioutil"
	"testing"
)

func TestProcess(t *testing.T) {

	// Test message "stats" parsing
	// Mock socket response
	b, err := ioutil.ReadFile("message.json")
	if err != nil {
		println(err)
		return
	}

	reply := process(b)

	//handle each message accordingly
	messageType := reply["message"].(string)
	println("Received message: " + messageType)

	if messageType == "stats" {
		if reply["cels"] != nil {
			cells := reply["cells"].(map[string]interface{})
			// cells iteration
			for key, value := range cells {
				println("cell " + key + "\n")
				// cell values iteration
				cell := value.(map[string]interface{})
				for k, v := range cell {
					switch k {
					case "dl_bitrate":
						println(v.(float64))
					case "ul_bitrate":
						println(v.(float64))
					case "dl_tx":
						println(v.(float64))
					case "ul_tx":
						println(v.(float64))
					}
				}
			}
		} else {
			println("empty empty")
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
