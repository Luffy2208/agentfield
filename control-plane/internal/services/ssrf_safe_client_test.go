package services

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIsPrivateIP(t *testing.T) {
	tests := []struct {
		name    string
		ip      string
		private bool
	}{
		// Loopback
		{"ipv4 loopback", "127.0.0.1", true},
		{"ipv4 loopback high", "127.255.255.255", true},
		{"ipv6 loopback", "::1", true},

		// Link-local
		{"ipv4 link-local", "169.254.0.1", true},
		{"ipv4 link-local metadata", "169.254.169.254", true},
		{"ipv6 link-local", "fe80::1", true},

		// RFC-1918
		{"10.x.x.x", "10.0.0.1", true},
		{"10.x.x.x high", "10.255.255.255", true},
		{"172.16.x.x", "172.16.0.1", true},
		{"172.31.x.x", "172.31.255.255", true},
		{"192.168.x.x", "192.168.0.1", true},
		{"192.168.x.x high", "192.168.255.255", true},

		// Unspecified
		{"ipv4 unspecified", "0.0.0.0", true},
		{"ipv6 unspecified", "::", true},

		// IPv6 ULA
		{"ipv6 ULA", "fd00::1", true},

		// Nil IP
		{"nil ip", "", true},

		// Public IPs (should NOT be private)
		{"public 8.8.8.8", "8.8.8.8", false},
		{"public 1.1.1.1", "1.1.1.1", false},
		{"public 93.184.216.34", "93.184.216.34", false},
		{"public ipv6", "2001:db8::1", false},
		{"172.32 (just outside RFC-1918)", "172.32.0.1", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var ip net.IP
			if tt.ip != "" {
				ip = net.ParseIP(tt.ip)
			}
			got := isPrivateIP(ip)
			assert.Equal(t, tt.private, got, "isPrivateIP(%s)", tt.ip)
		})
	}
}

func TestIsPrivateHost(t *testing.T) {
	tests := []struct {
		host    string
		private bool
	}{
		{"localhost", true},
		{"LOCALHOST", true},
		{"foo.localhost", true},
		{"sub.localhost", true},
		{"example.com", false},
		{"notlocalhost", false},
		{"localhost.example.com", false},
	}

	for _, tt := range tests {
		t.Run(tt.host, func(t *testing.T) {
			assert.Equal(t, tt.private, isPrivateHost(tt.host))
		})
	}
}

func TestValidateWebhookURL(t *testing.T) {
	tests := []struct {
		name      string
		url       string
		wantError bool
		errSubstr string
	}{
		// Should be rejected
		{"loopback ipv4", "http://127.0.0.1:9999/cb", true, "private/internal address"},
		{"cloud metadata", "http://169.254.169.254/latest/meta-data/", true, "private/internal address"},
		{"rfc1918 10.x", "http://10.0.0.1:8080/cb", true, "private/internal address"},
		{"rfc1918 192.168.x", "http://192.168.1.1:8080/cb", true, "private/internal address"},
		{"rfc1918 172.16.x", "http://172.16.0.1:8080/cb", true, "private/internal address"},
		{"localhost", "http://localhost:9999/cb", true, "private/internal host"},
		{"ipv6 loopback", "http://[::1]:9999/cb", true, "private/internal address"},
		{"unspecified", "http://0.0.0.0:9999/cb", true, "private/internal address"},
		{"subdomain localhost", "http://foo.localhost:9999/cb", true, "private/internal host"},
		{"empty host", "http:///path", true, "no host"},

		// Should be allowed
		{"public domain", "https://example.com/cb", false, ""},
		{"public ip", "http://93.184.216.34:8080/cb", false, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateWebhookURL(tt.url)
			if tt.wantError {
				require.Error(t, err, "expected error for %s", tt.url)
				assert.Contains(t, err.Error(), tt.errSubstr)
			} else {
				assert.NoError(t, err, "expected no error for %s", tt.url)
			}
		})
	}
}

func TestNewSSRFSafeClient_BlocksPrivateIPs(t *testing.T) {
	// Start a test server on loopback — the SSRF-safe client should refuse to connect.
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"secret": "leaked"}`))
	}))
	defer ts.Close()

	client := NewSSRFSafeClient(5e9) // 5s timeout
	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, ts.URL, nil)
	require.NoError(t, err)

	_, err = client.Do(req)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "private/internal address")
}

func TestNewSSRFSafeClient_AllowsPublicIPs(t *testing.T) {
	// We can't easily test a real public IP in CI, but we can verify the client
	// is constructed without error and has a non-nil transport.
	client := NewSSRFSafeClient(5e9)
	require.NotNil(t, client)
	require.NotNil(t, client.Transport)
}
