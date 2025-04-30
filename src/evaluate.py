import os
import numpy as np
import yaml
import json
import mlflow
import mlflow.keras
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def evaluate_model():
    params = load_params()
    batch_size = params['evaluation']['batch_size']
    
    data_dir = Path('data/processed')
    metrics_dir = Path('metrics')
    metrics_dir.mkdir(exist_ok=True)
    
    X_test = np.load(data_dir / 'X_test.npy')
    y_test = np.load(data_dir / 'y_test.npy')
    
    model = tf.keras.models.load_model('models/model.h5')
    
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("emotion_recognition")
    
    with mlflow.start_run():
        test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)
        
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        metrics = {
            "accuracy": float(test_accuracy),
            "loss": float(test_loss)
        }
        
        emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        report = classification_report(y_true_classes, y_pred_classes, 
                                      target_names=emotion_labels, output_dict=True)
        
        metrics.update({
            "precision": report['weighted avg']['precision'],
            "recall": report['weighted avg']['recall'],
            "f1_score": report['weighted avg']['f1-score']
        })
        
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1_score", metrics["f1_score"])
        
        with open(metrics_dir / 'evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f)

if __name__ == "__main__":
    evaluate_model()