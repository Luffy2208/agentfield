package packages

import "os"

func resolveServerURL() string {
	if v := os.Getenv("AGENTFIELD_SERVER"); v != "" {
		return v
	}
	if v := os.Getenv("AGENTFIELD_SERVER_URL"); v != "" {
		return v
	}
	return "http://localhost:8080"
}
