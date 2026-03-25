#!/bin/bash
# Auto-restart daemon — single instance only (PID lock)
cd /home/asepyudi/saham-scanner

PIDFILE="/tmp/saham_scanner.pid"

# Kill any existing instances
if [ -f "$PIDFILE" ]; then
    OLD_PID=$(cat "$PIDFILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Killing existing daemon PID $OLD_PID..."
        kill "$OLD_PID"
        sleep 2
    fi
    rm -f "$PIDFILE"
fi

# Also kill any stray main.py processes
pkill -f "saham-scanner/main.py" 2>/dev/null
sleep 2

echo $$ > "$PIDFILE"

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting saham-scanner daemon..."
    .venv/bin/python -u main.py >> /tmp/saham_scanner.log 2>&1
    EXIT_CODE=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Daemon exited with code $EXIT_CODE — restarting in 10s..."
    sleep 10
done
