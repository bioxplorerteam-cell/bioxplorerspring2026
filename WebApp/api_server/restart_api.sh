#!/bin/bash
# Script to restart the API server

echo "Stopping API server..."
pkill -f "uvicorn app:app"
sleep 2

echo "Starting API server..."
cd /p/realai/BioXplorer/LLama-BioXplorer/WebApp/api_server
nohup python -m uvicorn app:app --host 0.0.0.0 --port 5001 > api_server.log 2>&1 &

echo "API server restarting... Check api_server.log for status"
echo "Use 'tail -f api_server.log' to monitor startup"
