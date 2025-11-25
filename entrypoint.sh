#!/bin/sh
# Exit immediately if a command exits with a non-zero status.
set -e
echo "Starting Uvicorn server..."
uvicorn main:app --reload --host 0.0.0.0 --port 8000
