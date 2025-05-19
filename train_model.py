"""
Model training module.
Contains functions for training and evaluating the models for frequency and CM prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from typing import Tuple, Dict, Any
import pickle
from sklearn.ensemble import RandomForestRegressor
import time

from config import (
    TEST_SIZE, RANDOM_STATE, 
    MODEL_FREQ_PATH, MODEL_CM_PATH,
    X_TRAIN_PATH, Y_TRAIN_PATH,
    N_JOBS
)
from preprocessing import load_data, preprocess_train

def train_single_model(x_train: pd.DataFrame, y_train: pd.Series, model_path: str, model_name: str) -> None:
    """
    Train and save a prediction model.
    
    Args:
        x_train: Training features
        y_train: Training target
        model_path: Path to save the model
        model_name: Name of the model for logging
    """
    print(f"Training {model_name} model...")
    start_time = time.time()
    
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=N_JOBS,
        verbose=0
    )
    model.fit(x_train, y_train)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    end_time = time.time()
    print(f"{model_name} model trained in {end_time - start_time:.2f} seconds")

def train():
    """
    Main training function for frequency and CM prediction models.
    """
    print("Loading data...")
    start_time = time.time()
    
    # Load and preprocess data
    x_train, y_train, _ = load_data(X_TRAIN_PATH, Y_TRAIN_PATH, X_TRAIN_PATH)
    x_train_prep = preprocess_train(x_train)
    
    print(f"Data loaded and preprocessed in {time.time() - start_time:.2f} seconds")
    
    # Prepare training data
    y_freq = y_train['FREQ']
    y_cm = y_train['CM']
    
    # Train models
    train_single_model(x_train_prep, y_freq, MODEL_FREQ_PATH, "frequency")
    train_single_model(x_train_prep, y_cm, MODEL_CM_PATH, "CM")
    
    print('Models trained and saved successfully.')

if __name__ == "__main__":
    train()
