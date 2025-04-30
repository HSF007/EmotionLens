#!/bin/bash
set -e

# Copy dashboard.json to the provisioning dashboards directory
cp /app/monitoring/grafana/dashboard.json /etc/grafana/provisioning/dashboards/

# Set permissions
chmod 644 /etc/grafana/provisioning/dashboards/dashboard.json