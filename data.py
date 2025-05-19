import pandas as pd
import json


df = pd.read_csv("ton_fichier.csv")

cols_to_drop = ['CARACT2', 'CARACT3', 'TYPBAT1', 'DEROG12', 'DEROG13', 'DEROG14', 'DEROG16']
df = df.drop(columns=cols_to_drop, errors='ignore')  # ignore si colonne absente

row = df.iloc[0].to_dict()


payload = {
    "data": [row]
}

json_payload = json.dumps(payload, indent=2, ensure_ascii=False)
print(json_payload)

