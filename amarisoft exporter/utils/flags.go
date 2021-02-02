package utils

import (
	"flag"
)

// Config will be the holder for our flags
type Config struct {
	path string
}

// Setup initializes a config from flags passed inn
func (c *Config) Setup() {

	flag.StringVar(&c.path, "config", "", "path of config file")
}

//GetPath returns private path value
func (c *Config) GetPath() string {
	msg := c.path
	return msg
}
