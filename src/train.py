import os
import numpy as np
import yaml
import json
import mlflow
import mlflow.keras
from pathlib import Path
from model import create_emotion_model
import tensorflow as tf

def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def train_model():
    params = load_params()
    img_size = params['preprocessing']['image_size']
    learning_rate = params['training']['learning_rate']
    batch_size = params['training']['batch_size']
    epochs = params['training']['epochs']
    dropout_rate = params['training']['dropout_rate']
    
    data_dir = Path('data/processed')
    models_dir = Path('models')
    metrics_dir = Path('metrics')
    
    models_dir.mkdir(exist_ok=True)
    metrics_dir.mkdir(exist_ok=True)
    
    X_train = np.load(data_dir / 'X_train.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    
    X_val = np.load(data_dir / 'X_validation.npy')
    y_val = np.load(data_dir / 'y_validation.npy')
    
    num_classes = params['training']['num_classes']
    input_shape = (img_size, img_size, 1)
    
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("emotion_recognition")
    
    with mlflow.start_run():
        model = create_emotion_model(input_shape, num_classes, dropout_rate)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("dropout_rate", dropout_rate)
        
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        for epoch in range(epochs):
            mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
        
        model.save(models_dir / 'model.h5')
        mlflow.keras.log_model(model, "model")
        
        metrics = {
            "train_accuracy": float(history.history['accuracy'][-1]),
            "val_accuracy": float(history.history['val_accuracy'][-1]),
            "train_loss": float(history.history['loss'][-1]),
            "val_loss": float(history.history['val_loss'][-1])
        }
        
        with open(metrics_dir / 'train_metrics.json', 'w') as f:
            json.dump(metrics, f)

if __name__ == "__main__":
    train_model()