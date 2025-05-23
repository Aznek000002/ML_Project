"""
Main application module.
Contains the FastAPI application and endpoints.
"""

from fastapi import FastAPI, HTTPException
import pandas as pd
import pickle
from pydantic import BaseModel
from typing import List, Dict, Any

from preprocessing import preprocess_test
from config import MODEL_FREQ_PATH, MODEL_CM_PATH

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Prédiction de la charge Sinistres API",
    description="API pour prédire la charge des sinistres en assurance",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement des modèles
try:
    with open(MODEL_FREQ_PATH, 'rb') as f:
        model_freq = pickle.load(f)
    with open(MODEL_CM_PATH, 'rb') as f:
        model_cm = pickle.load(f)
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {str(e)}")
    raise

# ✅ Schémas de requête et réponse
class PredictionRequest(BaseModel):
    data: List[Dict[str, Any]]

class PredictionResponse(BaseModel):
    predictions_freq: List[float]
    predictions_cm: List[float]

class PredictionMontantResponse(BaseModel):
    predictions_montant: List[float]

# Endpoint de santé
@app.get("/health")
async def health_check():
    try:
        if model_freq is None or model_cm is None:
            return {"status": "unhealthy", "error": "Model not loaded"}
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# Endpoint principal
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        df = pd.DataFrame(request.data)
        df_prep = preprocess_test(df)
        predictions_freq = model_freq.predict(df_prep)
        predictions_cm = model_cm.predict(df_prep)
        return PredictionResponse(
            predictions_freq=predictions_freq.tolist(),
            predictions_cm=predictions_cm.tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ Nouveau endpoint pour le montant
@app.post("/predict_montant", response_model=PredictionMontantResponse)
async def predict_montant(request: PredictionRequest):
    try:
        df = pd.DataFrame(request.data)
        df_prep = preprocess_test(df)
        freq_preds = model_freq.predict(df_prep)
        cm_preds = model_cm.predict(df_prep)
        montant_preds = freq_preds * cm_preds
        return PredictionMontantResponse(predictions_montant=montant_preds.tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Lancement local
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
