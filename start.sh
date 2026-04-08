#!/bin/bash
# HF Spaces runs on port 7860. OpenEnv validator expects port 8000.
# Use socat to forward 8000 -> 7860 so both work.
APP_PORT=${PORT:-7860}
if [ "$APP_PORT" != "8000" ]; then
    socat TCP-LISTEN:8000,fork,reuseaddr TCP:localhost:${APP_PORT} &
fi
exec uvicorn server.app:app --host 0.0.0.0 --port ${APP_PORT} --workers 1
