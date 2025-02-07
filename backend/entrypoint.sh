#!/bin/bash
set -e

echo "Creating database migrations..."
python manage.py makemigrations inventory forecasting file projects users clients

echo "Applying database migrations..."
python manage.py migrate

echo "Starting Gunicorn..."
exec gunicorn --workers 8 --timeout 86400 --access-logfile - --log-level debug main.wsgi:application --bind 0.0.0.0:8000
