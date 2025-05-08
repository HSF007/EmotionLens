# EmotionLens



## Features

- Data versioning with DVC
- Automated ML pipeline for data preprocessing, model training, and evaluation
- Experiment tracking with MLflow
- Model deployment via FastAPI
- Monitoring with Prometheus and Grafana

## Project Structure

```
EmotionLens/
├── data/               # Data directory (tracked by DVC)
├── src/                # Source code
├── app/                # API application
├── monitoring/         # Monitoring configuration
├── models/             # Model storage
├── notebooks/          # Jupyter notebooks
├── dvc.yaml            # DVC pipeline configuration
├── params.yaml         # Parameters for the pipeline
├── requirements.txt    # Python dependencies
└── Dockerfile          # Container definition
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/EmotionLens.git
cd EmotionLens
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Initialize DVC:
```bash
dvc init
```

## Data Structure

Place your emotion dataset in the `data` directory with the following structure:

```
data/
├── train/
│   ├── anger/
│   ├── disgust/
│   ├── fear/
│   ├── happiness/
│   ├── sadness/
│   ├── surprise/
│   └── neutral/
├── test/
└── validation/
```

Each emotion folder should contain facial images displaying that emotion.

## Running the Pipeline

Execute the entire ML pipeline:

```bash
dvc repro
```

This will run:
1. Data preprocessing
2. Model training
3. Model evaluation
4. Deployment of the best model

## Viewing Experiments

Start the MLflow UI:

```bash
mlflow ui
```

Visit http://localhost:5000 to view experiment results.

## Running the API

Start the FastAPI application:

```bash
cd app
uvicorn main:app --reload
```

The API will be available at http://localhost:8000

## Monitoring

1. Start Prometheus:
```bash
docker run -d --name prometheus -p 9090:9090 -v $(pwd)/monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus
```

2. Start Grafana:
```bash
docker run -d --name grafana -p 3000:3000 grafana/grafana
```

3. Import the dashboard from `monitoring/grafana/dashboard.json`

Visit http://localhost:3000 to access the Grafana dashboard.

## Using Docker

The entire application is containerized using Docker Compose, which sets up:
- The EmotionLens application
- Prometheus for metrics collection
- Grafana for monitoring dashboards

### Option 1: Quick Start with Docker Compose

1. Run the complete pipeline:
```bash
chmod +x run-pipeline.sh
./run-pipeline.sh
```

This script will:
- Build the Docker image
- Run data preprocessing, model training, and evaluation
- Start all services

2. Access the services:
- EmotionLens API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (login with admin/admin)

### Option 2: Manual Docker Setup

1. Build and start the services:
```bash
docker-compose build
docker-compose up -d
```

2. Run the ML pipeline:
```bash
docker-compose exec app python src/data_preprocessing.py
docker-compose exec app python src/train.py
docker-compose exec app python src/evaluate.py
```

### Stopping the Application

```bash
docker-compose down
```

### Persisted Data

The following data is persisted through Docker volumes:
- ML models: `./models` directory
- Training data: `./data` directory
- MLflow tracking: `./mlruns` directory
- Prometheus data: Docker volume `prometheus-data`
- Grafana data: Docker volume `grafana-data`

## High level Design (HLD)

```mermaid
graph TD
    UI[User Interface - FastAPI] --> Upload[Upload Endpoint - predict]
    Upload --> Preprocessing[Preprocess Image]
    Preprocessing --> Model[Emotion Detection Model - CNN]
    Model --> Output[Return Emotion & Confidence]
    Model --> Metrics[Prometheus Metrics]
    UI --> Eval[Metrics Endpoint - metrics]
    
    subgraph Training_Pipeline
        RawData[Raw Dataset] --> PreprocessTrain[Preprocess Data - data_preprocessing.py]
        PreprocessTrain --> Train[Train Model - train.py]
        Train --> SaveModel[Save model.h5]
        Train --> LogTrain[Log to MLflow]
        SaveModel --> Model
        Train --> Evaluate[Evaluate Model - evaluate.py]
        Evaluate --> Promote[Model Promotion - temp.py]
    end
```

## Low Level Design (LLD)

```mermaid
graph TD
    %% FastAPI Application
    subgraph "FastAPI App (main.py)"
        A1[POST /predict] --> A2[Read and Decode Image]
        A2 --> A3[Call predict_emotion]
        A3 --> A4[Return JSON with Emotion & Confidence]
        A4 --> A5[Log Prediction]
        A4 --> A6[Prometheus Metrics Update]
    end
    
    %% Prediction Pipeline
    subgraph "Prediction Pipeline (predict.py)"
        B1[predict_emotion]
        B2[Load model from best_model/]
        B3[Convert image to grayscale]
        B4[Resize and normalize image]
        B5[Predict using model]
        B6[Return emotion class and confidence]
        B1 --> B2 --> B3 --> B4 --> B5 --> B6
    end
    
    %% Training Process
    subgraph "Training (train.py)"
        C1[Load Train & Validation Data]
        C2[Build Model using model.py]
        C3[Compile Model]
        C4[Train Model with Callbacks]
        C5[Log metrics to MLflow]
        C6[Save Trained Model to artifacts/]
        C1 --> C2 --> C3 --> C4 --> C5 --> C6
    end
    
    %% Evaluation Process
    subgraph "Evaluation (evaluate.py)"
        D1[Load Test Data]
        D2[Load Trained Model from artifacts/]
        D3[Evaluate and Generate Predictions]
        D4[Create Classification Report]
        D5[Log metrics to MLflow]
        D6[Save metrics to evaluation_metrics.json]
        D1 --> D2 --> D3 --> D4 --> D5 --> D6
    end
    
    %% Model Promotion
    subgraph "Model Promotion (promote.py)"
        E1[Read evaluation_metrics.json]
        E2{Accuracy > 0.60?}
        E3[Copy model.h5 to best_model/]
        E4[Keep current production model]
        E1 --> E2
        E2 -->|Yes| E3
        E2 -->|No| E4
    end
    
    %% Connections between components
    A3 -.-> B1
    C6 -.-> D2
    D6 -.-> E1
    E3 -.-> B2
```