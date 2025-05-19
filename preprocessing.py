"""
Data preprocessing module.
Contains functions for cleaning and preparing the data for model training.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List, Dict
from config import NAN_THRESHOLD, COLS_TO_DROP_PATH, ENCODERS_PATH, CHUNK_SIZE, DTYPE_OPTIMIZATION

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by converting dtypes.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with optimized dtypes
    """
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

def load_data(x_train_path: str, y_train_path: str, x_test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the training and test datasets with optimized memory usage.
    
    Args:
        x_train_path: Path to X training data
        y_train_path: Path to y training data
        x_test_path: Path to X test data
        
    Returns:
        Tuple containing (X_train, y_train, X_test)
    """
    # Load data in chunks if file is large
    x_train_chunks = []
    for chunk in pd.read_csv(x_train_path, chunksize=CHUNK_SIZE, low_memory=False):
        if DTYPE_OPTIMIZATION:
            chunk = optimize_dtypes(chunk)
        x_train_chunks.append(chunk)
    x_train = pd.concat(x_train_chunks, axis=0)
    
    y_train = pd.read_csv(y_train_path)
    if DTYPE_OPTIMIZATION:
        y_train = optimize_dtypes(y_train)
    
    x_test_chunks = []
    for chunk in pd.read_csv(x_test_path, chunksize=CHUNK_SIZE, low_memory=False):
        if DTYPE_OPTIMIZATION:
            chunk = optimize_dtypes(chunk)
        x_test_chunks.append(chunk)
    x_test = pd.concat(x_test_chunks, axis=0)
    
    return x_train, y_train, x_test

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with handled missing values
    """
    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Fill categorical columns with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    
    return df

def drop_cols_with_many_nans(df, threshold=NAN_THRESHOLD):
    nan_ratio = df.isna().mean()
    cols_to_drop = nan_ratio[nan_ratio > threshold].index.tolist()
    df_cleaned = df.drop(columns=cols_to_drop)
    print(f"{len(cols_to_drop)} colonne(s) supprimée(s) : {cols_to_drop}")
    return df_cleaned, cols_to_drop

def save_cols_to_drop(cols_to_drop, path=COLS_TO_DROP_PATH):
    with open(path, 'wb') as f:
        pickle.dump(cols_to_drop, f)

def load_cols_to_drop(path=COLS_TO_DROP_PATH):
    with open(path, 'rb') as f:
        return pickle.load(f)

def encode_categorical(df, encoders=None, fit=True):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if encoders is None:
        encoders = {}
    for col in cat_cols:
        if fit:
            le = LabelEncoder()
            df[col] = df[col].astype(str).fillna('NA')
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le is not None:
                df[col] = df[col].astype(str).fillna('NA')
                df[col] = le.transform(df[col])
            else:
                df[col] = df[col].astype(str).fillna('NA')
    return df, encoders

def save_encoders(encoders, path=ENCODERS_PATH):
    with open(path, 'wb') as f:
        pickle.dump(encoders, f)

def load_encoders(path=ENCODERS_PATH):
    with open(path, 'rb') as f:
        return pickle.load(f)

def preprocess_train(x_train):
    """
    Optimized preprocessing pipeline for training data.
    """
    # Drop columns with too many NaNs
    x_train_clean, cols_to_drop = drop_cols_with_many_nans(x_train)
    save_cols_to_drop(cols_to_drop)
    
    # Fill remaining NaNs efficiently
    x_train_clean = x_train_clean.fillna(-1)
    
    # Optimize memory before encoding
    if DTYPE_OPTIMIZATION:
        x_train_clean = optimize_dtypes(x_train_clean)
    
    # Encode categorical variables
    x_train_encoded, encoders = encode_categorical(x_train_clean, fit=True)
    save_encoders(encoders)
    
    return x_train_encoded

def preprocess_test(x_test):
    # Load columns to drop
    cols_to_drop = load_cols_to_drop()
    x_test_clean = x_test.drop(columns=cols_to_drop, errors='ignore')
    x_test_clean = x_test_clean.fillna(-1)
    # Load encoders
    encoders = load_encoders()
    x_test_encoded, _ = encode_categorical(x_test_clean, encoders=encoders, fit=False)
    return x_test_encoded

def preprocess_data(x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Main preprocessing function that handles all data preparation steps.
    
    Args:
        x_train: Training features
        y_train: Training targets
        x_test: Test features
        
    Returns:
        Tuple containing (processed X_train, y_train, X_test, encoders)
    """
    # Handle missing values
    x_train = handle_missing_values(x_train)
    x_test = handle_missing_values(x_test)
    
    # Encode categorical features
    x_train, encoders = encode_categorical(x_train, fit=True)
    x_test, _ = encode_categorical(x_test, fit=False)
    
    return x_train, y_train, x_test, encoders
