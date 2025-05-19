# Projet de Prédiction Fréquence Sinistres

Ce projet est une application de machine learning pour prédire la fréquence des sinistres en assurance. Il est basé sur un notebook Jupyter existant et a été adapté pour la production.

## Structure du Projet

```
.
├── config.py           # Configuration et chemins
├── preprocessing.py    # Prétraitement des données
├── train_model.py      # Entraînement du modèle
├── predict.py          # Prédiction batch
├── main.py            # API FastAPI
├── requirements.txt    # Dépendances
└── README.md          # Documentation
```

## Installation

1. Cloner le repository
2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Configuration

Les chemins des fichiers et les paramètres sont configurés dans `config.py` :
- Chemins des données d'entraînement et de test
- Chemins des modèles sauvegardés
- Paramètres de prétraitement (seuil de NaN, etc.)

## Préprocessing

Le module `preprocessing.py` contient les fonctions de prétraitement :
- Suppression des colonnes avec trop de valeurs manquantes (> 90%)
- Remplissage des valeurs manquantes restantes
- Encodage des variables catégorielles
- Sauvegarde des encodeurs et de la liste des colonnes supprimées

## Entraînement

Pour entraîner le modèle :
```bash
python train_model.py
```

Le script :
1. Charge les données
2. Applique le prétraitement
3. Entraîne un modèle RandomForest pour la prédiction de la fréquence
4. Sauvegarde le modèle

## Prédiction Batch

Pour faire des prédictions sur un jeu de données :
```bash
python predict.py
```

Le script :
1. Charge les données de test
2. Applique le même prétraitement que l'entraînement
3. Charge le modèle sauvegardé
4. Génère les prédictions de fréquence
5. Sauvegarde les résultats dans un fichier CSV

## API

L'API FastAPI permet de faire des prédictions en temps réel :

1. Démarrer l'API :
```bash
python main.py
```

2. Endpoints disponibles :
- `POST /predict` : Prédiction de la fréquence des sinistres
- `GET /health` : Vérification de l'état de l'API

Exemple de requête :
```python
import requests
import json

data = [
    {
        "ACTIVIT2": "ACT1",
        "VOCATION": "VOC6",
        # ... autres features ...
    }
]

response = requests.post(
    "http://localhost:8000/predict",
    json={"data": data}
)
predictions = response.json()
```

## Format des Données

### Entrée
- Les données doivent contenir les mêmes colonnes que le jeu d'entraînement
- Les variables catégorielles seront automatiquement encodées
- Les valeurs manquantes seront gérées selon le pipeline de prétraitement

### Sortie
```json
{
    "predictions": [0.123, 0.456, ...]  # Liste des prédictions de fréquence
}
```

## Dépendances

Les dépendances principales sont listées dans `requirements.txt` :
- pandas
- numpy
- scikit-learn
- fastapi
- uvicorn
- pickle5
