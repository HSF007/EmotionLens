import cv2
import os
import numpy as np
import tensorflow as tf
import yaml
# from pathlib import Path

def load_params():
    par_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(par_dir, "params.yaml")
    with open(file_path, 'r') as f:
        params = yaml.safe_load(f)
    return params

def load_model():
    par_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(par_dir, "models/best_model/model.h5")
    # model_path = Path('models/best_model/model.h5')
    if not os.path.exists(model_path):
        model_path = os.path.join(par_dir, "models/model.h5")
        # model_path = Path('EmotionLens/models/model.h5')
        
    return tf.keras.models.load_model(str(model_path))

def preprocess_image(image, img_size):
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image = cv2.resize(image, (img_size, img_size))
    image = image / 255.0
    image = image.reshape(1, img_size, img_size, 1)
    return image

def predict_emotion(image):
    params = load_params()
    img_size = params['preprocessing']['image_size']
    
    model = load_model()
    
    preprocessed_image = preprocess_image(image, img_size)
    
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    prediction = model.predict(preprocessed_image)
    emotion_idx = np.argmax(prediction)
    emotion = emotion_labels[emotion_idx]
    confidence = float(prediction[0][emotion_idx])
    
    return {
        'emotion': emotion,
        'confidence': confidence,
        'all_probabilities': {label: float(prob) for label, prob in zip(emotion_labels, prediction[0])}
    }