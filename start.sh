#!/bin/bash
set -e

PORT=${PORT:-3000}
BACKEND_PORT=8000

# Start FastAPI backend
echo "Starting FastAPI backend on port $BACKEND_PORT..."
python -m uvicorn backend.main:app --host 0.0.0.0 --port $BACKEND_PORT &

# Wait for backend
sleep 3

# Start Next.js (rewrites proxy /api/* → FastAPI)
echo "Starting Next.js on port $PORT..."
NEXT_PUBLIC_API_URL=http://localhost:$BACKEND_PORT npx next start -p $PORT
