import os
import cv2
import numpy as np
import yaml
import tensorflow as tf
from pathlib import Path

def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def preprocess_data():
    params = load_params()
    img_size = params['preprocessing']['image_size']
    
    data_dir = Path('data')
    processed_dir = data_dir / 'processed'
    processed_dir.mkdir(exist_ok=True)
    
    for split in ['train', 'test', 'validation']:
        dataset_dir = data_dir / 'raw'
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
            
        X_data = []
        y_data = []
        
        emotion_labels = {
            'angry': 0, 
            'disgust': 1, 
            'fear': 2, 
            'happy': 3, 
            'sad': 4, 
            'surprise': 5, 
            'neutral': 6
        }
        
        for emotion in emotion_labels.keys():
            emotion_dir = split_dir / emotion
            if not emotion_dir.exists():
                continue
                
            for img_path in emotion_dir.glob('*.jpg'):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                    
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                img = cv2.resize(img, (img_size, img_size))
                img = img / 255.0
                
                X_data.append(img)
                y_data.append(emotion_labels[emotion])
        
        if X_data and y_data:
            X = np.array(X_data)
            X = X.reshape(X.shape[0], img_size, img_size, 1)
            y = tf.keras.utils.to_categorical(y_data, num_classes=len(emotion_labels))
            
            np.save(processed_dir / f'X_{split}.npy', X)
            np.save(processed_dir / f'y_{split}.npy', y)
            
            print(f"Processed {split} data: {X.shape[0]} images")

if __name__ == "__main__":
    preprocess_data()