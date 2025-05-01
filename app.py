## Cmd Terminal
"""
cd "H:\Desktop\Challenge Data"
uvicorn app:app --reload
"""

# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import pandas as pd
import pickle
import tempfile
import os

app = FastAPI( 
    title="🚜 API Prédiction Assurance - Crédit Agricole Challenge",
    description="""

---

🏆 **Bienvenue sur notre API officielle du Challenge Data !**

Cette API vous permet de générer des prédictions pour participer au challenge **InsurPrime: Can You Guess the Insurance Premium?**, organisé par Crédit Agricole Assurances.

## 🚀 Fonctionnalités

- `/download_sample` : Télécharger un fichier CSV **exemple** prêt à l’emploi.
- `/predict_freq` : Prédire la fréquence des sinistres (FREQ),
- `/predict_montant` : Prédire le montant moyen des sinistres (CM),
- `/predict_global` : Générer le fichier complet (FREQ + CM + CHARGE) ➔ Téléchargeable au format CSV.

## 📂 Format attendu

Un fichier CSV contenant vos données d'entrée (exemple : `test_input_for_API.csv`).

## ✅ Comment utiliser

1. Rendez-vous sur `/docs` (Swagger UI),
2. Utilisez `/download_sample` pour récupérer un modèle de CSV si besoin,
3. Choisissez la route souhaitée,
4. Uploadez votre fichier CSV via le bouton **"Try it out"**,
5. Cliquez sur **"Execute"** pour lancer la prédiction,
6. **Pour `/predict_global` : descendez et cliquez sur le bouton `Download file` pour récupérer votre fichier CSV à envoyer**.

## 🔗 Soumission officielle

Le fichier CSV généré par `/predict_global` est le fichier **à déposer directement** sur la plateforme du challenge ici :

👉 [Soumettre votre fichier sur Challenge Data](https://challengedata.ens.fr/participants/challenges/161/submissions)

---
""",
    version="1.0.0"
)

# 📦 Charger les modèles une seule fois au démarrage
with open("model_freq.pkl", "rb") as f:
    model_freq = pickle.load(f)
with open("model_cm.pkl", "rb") as f:
    model_cm = pickle.load(f)


def preprocess_for_prediction(input_path: str):
    print("📥 Chargement du fichier à prédire...")
    X_new = pd.read_csv(input_path)

    if 'ANNEE_ASSURANCE_x' in X_new.columns and 'ANNEE_ASSURANCE_y' in X_new.columns:
        X_new.drop(columns=['ANNEE_ASSURANCE_y'], inplace=True)
        X_new.rename(columns={'ANNEE_ASSURANCE_x': 'ANNEE_ASSURANCE'}, inplace=True)

    print("🔄 Typage des colonnes...")
    for col in X_new.columns:
        try:
            X_new[col] = pd.to_numeric(X_new[col], errors='raise')
        except:
            X_new[col] = X_new[col].astype(str).str.strip().astype('category')

    if "ZONE" in X_new.columns:
        X_new["ZONE"] = X_new["ZONE"].astype(str).str.strip().astype("category")

    print(f"✅ Données prêtes : {X_new.shape}")
    return X_new


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/download_sample")
def download_sample():
    """
    📥 Télécharger le fichier CSV exemple déjà prêt (test_input_for_API.csv).
    """
    file_path = "test_input_for_API.csv"  # Chemin vers ton fichier existant
    return FileResponse(file_path, filename="test_input_for_API.csv", media_type="text/csv")

@app.post("/predict_freq")
async def predict_freq(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    X_to_predict = preprocess_for_prediction(tmp_path)
    y_pred_freq = model_freq.predict(X_to_predict)

    result = pd.DataFrame({
        "ID": X_to_predict["ID"],
        "FREQ": y_pred_freq
    })

    return result.to_dict(orient="records")


@app.post("/predict_montant")
async def predict_montant(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    X_to_predict = preprocess_for_prediction(tmp_path)
    y_pred_cm = model_cm.predict(X_to_predict)

    result = pd.DataFrame({
        "ID": X_to_predict["ID"],
        "CM": y_pred_cm
    })

    return result.to_dict(orient="records")


@app.post("/predict_global")
async def predict_global(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    X_to_predict = preprocess_for_prediction(tmp_path)
    y_pred_freq = model_freq.predict(X_to_predict)
    y_pred_cm = model_cm.predict(X_to_predict)

    df_submission = pd.DataFrame({
        "ID": X_to_predict["ID"],
        "FREQ": y_pred_freq,
        "CM": y_pred_cm,
        "ANNEE_ASSURANCE": X_to_predict["ANNEE_ASSURANCE"]
    })
    df_submission["CHARGE"] = df_submission["FREQ"] * df_submission["CM"] * df_submission["ANNEE_ASSURANCE"]

    output_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df_submission.to_csv(output_csv.name, index=False)

    return FileResponse(output_csv.name, filename="submission.csv", media_type="text/csv")

#http://127.0.0.1:8000
# http://127.0.0.1:8000/docs 
# http://127.0.0.1:8000/redoc