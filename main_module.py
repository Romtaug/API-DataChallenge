# Paste pre-Pyrfected Python
import numpy as np
import pandas as pd


def preprocess_all(
    path_train_input: str,
    path_test_input: str,
    path_train_output: str,
    n_train: int = 383610,
):
    """
    Prétraitement simplifié sans normalisation, uniquement typage et nettoyage.
    """
    print("📥 Chargement des fichiers...")
    test_input_train = pd.read_csv(path_train_input)
    test_input_real = pd.read_csv(path_test_input)
    train_output = pd.read_csv(path_train_output)

    print("🔗 Fusion des inputs...")
    train_input = pd.concat([test_input_train, test_input_real], ignore_index=True)

    print("🧬 Fusion des données sur 'ID' avec train_output...")
    df = train_input.merge(train_output, on="ID", how="left")

    if "ANNEE_ASSURANCE_x" in df.columns and "ANNEE_ASSURANCE_y" in df.columns:
        df.drop(columns=["ANNEE_ASSURANCE_y"], inplace=True)
        df.rename(columns={"ANNEE_ASSURANCE_x": "ANNEE_ASSURANCE"}, inplace=True)

    if "ZONE" in df.columns:
        df["ZONE"] = df["ZONE"].astype(str).str.strip().astype("category")

    print("🎯 Séparation features / cibles...")
    y = df[["FREQ", "CM"]].copy()
    columns_to_drop = ["FREQ", "CM", "CHARGE"]
    X = df.drop(columns=columns_to_drop, errors="ignore").copy()

    print("🔄 Conversion des colonnes...")
    for col in X.columns:
        try:
            X[col] = pd.to_numeric(X[col], errors="raise")
        except:
            X[col] = X[col].astype(str).str.strip().astype("category")

    print("✂️ Découpage train/test...")
    X_train = X.iloc[:n_train].copy()
    X_test = X.iloc[n_train:].copy()
    y_train_freq = y["FREQ"].iloc[:n_train].copy()
    y_train_cm = y["CM"].iloc[:n_train].copy()

    print("✅ Preprocessing terminé.")
    print(f" - X final : {X.shape}")
    print(f" - y final : {y.shape}")
    print(f" - X_train : {X_train.shape}")
    print(f" - X_test  : {X_test.shape}")
    print(f" - y_train_freq : {y_train_freq.shape}")
    print(f" - y_train_cm   : {y_train_cm.shape}")

    return X_train, X_test, y_train_freq, y_train_cm, df, X, y


X_train, X_test, y_train_freq, y_train_cm, df, X, y = preprocess_all(
    "train_input.csv", "test_input.csv", "train_output.csv"
)

print("\n📂 Colonnes finales utilisées (X.columns) :")
print(list(X.columns))

import warnings

import numpy as np
import optuna
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.utils._testing import ignore_warnings
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=ConvergenceWarning)


@ignore_warnings(category=ConvergenceWarning)
def train_model_freq(X_train, y_train_freq, X_test):
    print("🎯 Entraînement modèle FREQ avec Optuna")

    # Échantillonnage
    sample_idx = np.random.choice(X_train.index, size=50000, replace=False)
    X_sample = X_train.loc[sample_idx]
    y_sample = y_train_freq.loc[sample_idx]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150),
            "max_depth": trial.suggest_int("max_depth", 2, 4),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "random_state": 42,
            "enable_categorical": True,
            "objective": "reg:squarederror",
            "n_jobs": -1,
        }
        model = XGBRegressor(**params)
        return cross_val_score(
            model, X_sample, y_sample, cv=2, scoring="r2", n_jobs=-1
        ).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10, show_progress_bar=True)

    print(f"✅ Best params FREQ : {study.best_params}")
    model = XGBRegressor(
        **study.best_params,
        enable_categorical=True,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X_train, y_train_freq)

    # Sauvegarde
    with open("model_freq.pkl", "wb") as f:
        pickle.dump(model, f)

    y_pred = model.predict(X_test)
    return y_pred, model


################################################################################################


@ignore_warnings(category=ConvergenceWarning)
def train_model_cm(X_train, y_train_cm, X_test):
    print("🎯 Entraînement modèle CM avec Optuna")

    # Échantillonnage
    sample_size = min(50000, len(X_train))
    sample_idx = np.random.choice(X_train.index, size=sample_size, replace=False)
    X_sample = X_train.loc[sample_idx]
    y_sample = y_train_cm.loc[sample_idx]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "enable_categorical": True,
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": -1,
        }
        model = XGBRegressor(**params)
        return cross_val_score(
            model, X_sample, y_sample, cv=2, scoring="r2", n_jobs=-1
        ).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10, show_progress_bar=True)

    print(f"✅ Best params CM : {study.best_params}")
    model = XGBRegressor(
        **study.best_params,
        enable_categorical=True,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X_train, y_train_cm)

    # Sauvegarde
    with open("model_cm.pkl", "wb") as f:
        pickle.dump(model, f)

    y_pred = model.predict(X_test)
    return y_pred, model


y_pred_freq, model_freq = train_model_freq(X_train, y_train_freq, X_test)
y_pred_cm, model_cm = train_model_cm(X_train, y_train_cm, X_test)


import pickle

import pandas as pd


def preprocess_for_prediction(input_path: str):
    """
    Prétraitement léger pour prédiction (sans normalisation, avec typage).
    """
    print("📥 Chargement du fichier à prédire...")
    X_new = pd.read_csv(input_path)

    if "ANNEE_ASSURANCE_x" in X_new.columns and "ANNEE_ASSURANCE_y" in X_new.columns:
        X_new.drop(columns=["ANNEE_ASSURANCE_y"], inplace=True)
        X_new.rename(columns={"ANNEE_ASSURANCE_x": "ANNEE_ASSURANCE"}, inplace=True)

    print("🔄 Typage des colonnes...")
    for col in X_new.columns:
        try:
            X_new[col] = pd.to_numeric(X_new[col], errors="raise")
        except:
            X_new[col] = X_new[col].astype(str).str.strip().astype("category")

    if "ZONE" in X_new.columns:
        X_new["ZONE"] = X_new["ZONE"].astype(str).str.strip().astype("category")

    print(f"✅ Données prêtes : {X_new.shape}")
    return X_new


def generate_submission(input_path: str, output_path: str = "submission.csv"):
    """
    Fonction complète : chargement des modèles, prédiction sur données,
    calcul de CHARGE, et sauvegarde en CSV.
    """
    print("📦 Chargement des modèles...")
    with open("model_freq.pkl", "rb") as f:
        model_freq = pickle.load(f)
    with open("model_cm.pkl", "rb") as f:
        model_cm = pickle.load(f)

    print("🔍 Prétraitement des données...")
    X_to_predict = preprocess_for_prediction(input_path)

    print("📈 Prédictions en cours...")
    y_pred_freq = model_freq.predict(X_to_predict)
    y_pred_cm = model_cm.predict(X_to_predict)

    print("🧮 Construction du DataFrame de sortie...")
    df_submission = pd.DataFrame(
        {
            "ID": X_to_predict["ID"],
            "FREQ": y_pred_freq,
            "CM": y_pred_cm,
            "ANNEE_ASSURANCE": X_to_predict["ANNEE_ASSURANCE"],
        }
    )
    df_submission["CHARGE"] = (
        df_submission["FREQ"] * df_submission["CM"] * df_submission["ANNEE_ASSURANCE"]
    )

    print(f"💾 Sauvegarde dans '{output_path}'...")
    df_submission.to_csv(output_path, index=False)
    print("✅ Fichier généré avec succès.")


generate_submission("test_input.csv")


import pandas as pd

# Charge le fichier complet
df = pd.read_csv("test_input.csv")

# Prend seulement les 1000 premières lignes
df_small = df.head(1000)

# Sauvegarde dans un nouveau fichier
df_small.to_csv("test_input_for_API.csv", index=False)

print("✅ Nouveau fichier 'test_input_for_API.csv' créé avec 1000 lignes.")
