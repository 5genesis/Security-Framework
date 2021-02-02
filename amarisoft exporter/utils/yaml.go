package utils

import (
	"fmt"

	"github.com/go-yaml/yaml"
)

// YAMLData is our config struct
// with YAML struct tags
type YAMLData struct {
	Url      string `yaml:"amari_url"`
	Port     string `yaml:"port"`
	Interval string `yaml:"interval"`
}

//Decode will decode into YAMLData
func (t *YAMLData) Decode(data []byte) error {
	return yaml.Unmarshal(data, t)
}

//Unmarshall byte[] to yaml
func UnmarshalAll(configFile string) (YAMLData, error) {

	y := YAMLData{}

	if err := y.Decode([]byte(configFile)); err != nil {
		return y, err
	}
	fmt.Println("Config used =", y)
	return y, nil
}
