"""
Prediction module.
Contains functions for loading models and making predictions.
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any, Tuple
import pickle

from config import MODEL_FREQ_PATH, MODEL_MONTANT_PATH, X_TEST_PATH, MODEL_CM_PATH, PREDICTIONS_PATH
from preprocessing import handle_missing_values, encode_categorical_features, load_data, preprocess_test

class Predictor:
    """
    Class for making predictions using trained models.
    """
    
    def __init__(self):
        """
        Initialize the predictor by loading models and encoders.
        """
        self.freq_model = joblib.load(MODEL_FREQ_PATH)
        self.cm_model = joblib.load(MODEL_CM_PATH)
        self.encoders = joblib.load('model/encoders.pkl')
    
    def preprocess_input(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data for prediction.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        # Handle missing values
        data = handle_missing_values(data)
        
        # Encode categorical features
        data, _ = encode_categorical_features(data)
        
        return data
    
    def predict_freq(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict frequency values.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Array of frequency predictions
        """
        data = self.preprocess_input(data)
        return self.freq_model.predict(data)
    
    def predict_cm(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict CM (Confusion Matrix) values.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Array of CM predictions
        """
        data = self.preprocess_input(data)
        return self.cm_model.predict(data)

def load_predictor() -> Predictor:
    """
    Load and return a predictor instance.
    
    Returns:
        Initialized Predictor instance
    """
    return Predictor()

def predict():
    # Charger les données de test
    _, _, x_test = load_data(X_TEST_PATH, X_TEST_PATH, X_TEST_PATH)

    # Prétraitement (pipeline du notebook)
    x_test_prep = preprocess_test(x_test)

    # Charger les modèles
    with open(MODEL_FREQ_PATH, 'rb') as f:
        model_freq = pickle.load(f)
    with open(MODEL_CM_PATH, 'rb') as f:
        model_cm = pickle.load(f)

    # Prédictions
    freq_pred = model_freq.predict(x_test_prep)
    cm_pred = model_cm.predict(x_test_prep)

    # Sauvegarder les résultats
    predictions = pd.DataFrame({
        'FREQ_PRED': freq_pred,
        'CM_PRED': cm_pred
    })
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    print(f'Prédictions sauvegardées dans {PREDICTIONS_PATH}')

if __name__ == '__main__':
    predict()
