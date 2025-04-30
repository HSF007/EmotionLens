#!/bin/bash
set -e

echo "EmotionLens ML Pipeline"
echo "======================="

echo "1. Building Docker image..."
docker-compose build app

echo "2. Running data preprocessing..."
docker-compose run --rm app python src/data_preprocessing.py

echo "3. Training the model..."
docker-compose run --rm app python src/train.py

echo "4. Evaluating the model..."
docker-compose run --rm app python src/evaluate.py

echo "5. Starting the application..."
docker-compose up -d

echo "Setup complete! You can access:"
echo "- EmotionLens API: http://localhost:8000"
echo "- Prometheus: http://localhost:9090"
echo "- Grafana: http://localhost:3000 (admin/admin)"