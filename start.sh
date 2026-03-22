#!/bin/bash
# Auto-restart daemon jika crash
cd /home/asepyudi/saham-scanner

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting saham-scanner daemon..."
    .venv/bin/python -u main.py >> /tmp/saham_scanner.log 2>&1
    EXIT_CODE=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Daemon exited with code $EXIT_CODE — restarting in 10s..."
    sleep 10
done
