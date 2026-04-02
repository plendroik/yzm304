import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath="data/heart.csv", test_size=0.2, val_size=0.1, random_state=42):
    # kaggle api uzerinden veri indirme
    if not os.path.exists(filepath):
        try:
            with open('kaggle_credentials.json', 'r') as f:
                creds = json.load(f)
            os.environ['KAGGLE_USERNAME'] = creds.get('username')
            os.environ['KAGGLE_KEY'] = creds.get('key')
        except FileNotFoundError:
            raise Exception("kaggle_credentials.json bulunamadi.")
            
        try:
            import kaggle
            kaggle.api.authenticate()
            if not os.path.exists('data'):
                os.makedirs('data')
            kaggle.api.dataset_download_files('johnsmith109/heart-disease-dataset', path='data', unzip=True)
            if os.path.exists("data/heart.csv") and filepath == "data/heart_disease.csv":
                os.rename("data/heart.csv", filepath)
        except Exception as e:
            raise Exception(f"Kaggle hatasi: {e}")
            
    # hedef degiskenin ve ozelliklerin ayrilmasi
    df = pd.read_csv(filepath)
    X = df.drop('target', axis=1).values
    y = df['target'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # egitim, dogrulama ve test setlerinin olusturulmasi
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )
    
    return {
        'X_train': X_train, 'y_train': y_train.reshape(-1, 1),
        'X_val': X_val, 'y_val': y_val.reshape(-1, 1),
        'X_test': X_test, 'y_test': y_test.reshape(-1, 1),
        'scaler': scaler
    }
