"""
Placeholder exporter - to be built once the forgery detection model is
trained and served. This will expose:
  - inference latency
  - prediction confidence distribution
  - a simple data drift signal (e.g. comparing input feature stats to
    training-set stats)
via /metrics for Prometheus to scrape.
"""

import time

from prometheus_client import start_http_server, Gauge

placeholder = Gauge("docforge_exporter_up", "1 if the exporter is running")

if __name__ == "__main__":
    start_http_server(9100)
    placeholder.set(1)
    while True:
        time.sleep(3600)
