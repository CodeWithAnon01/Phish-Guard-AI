#!/bin/bash
echo "Starting PhishGuard AI Backend..."
cd backend
../venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload
