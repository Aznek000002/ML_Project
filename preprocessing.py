"""
Data preprocessing module.
Contains functions for cleaning and preparing the data for model training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List, Dict
from config import NAN_THRESHOLD, CHUNK_SIZE, DTYPE_OPTIMIZATION

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

def load_data(x_train_path: str, y_train_path: str, x_test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    
    return df

def drop_cols_with_many_nans(df, threshold=NAN_THRESHOLD):
    nan_ratio = df.isna().mean()
    cols_to_drop = nan_ratio[nan_ratio > threshold].index.tolist()
    df_cleaned = df.drop(columns=cols_to_drop)
    print(f"{len(cols_to_drop)} colonne(s) supprimée(s) : {cols_to_drop}")
    return df_cleaned, cols_to_drop

def encode_categorical(df, fit=True):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str).fillna('NA')
        if fit:
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            try:
                df[col] = le.transform(df[col])
            except:
                df[col] = le.fit_transform(df[col])
    return df, encoders

def preprocess_train(x_train):
    x_train_clean, _ = drop_cols_with_many_nans(x_train)
    x_train_clean = x_train_clean.fillna(-1)
    
    if DTYPE_OPTIMIZATION:
        x_train_clean = optimize_dtypes(x_train_clean)
    
    x_train_encoded, _ = encode_categorical(x_train_clean, fit=True)
    
    return x_train_encoded

def preprocess_test(x_test):
    # ✅ Colonnes supprimées pendant l'entraînement
    cols_to_drop = [
        "CARACT2", "CARACT3", "TYPBAT1",
        "DEROG12", "DEROG13", "DEROG14", "DEROG16"
    ]
    x_test_clean = x_test.drop(columns=cols_to_drop, errors='ignore')
    x_test_clean = x_test_clean.fillna(-1)

    # ✅ Encode directement sans .pkl
    x_test_encoded, _ = encode_categorical(x_test_clean, fit=True)

    return x_test_encoded

def preprocess_data(x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder]]:
    x_train = handle_missing_values(x_train)
    x_test = handle_missing_values(x_test)
    
    x_train, encoders = encode_categorical(x_train, fit=True)
    x_test, _ = encode_categorical(x_test, fit=True)
    
    return x_train, y_train, x_test, encoders
