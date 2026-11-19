"""Latih 6 model forecasting produksi padi (sama seperti notebook) dan simpan ke model_padi.joblib.

Jalankan:  python train_model.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

HERE = Path(__file__).parent
DATA = HERE / "Data_Tanaman_Padi_Sumatera_version_1.csv"
OUT = HERE / "model_padi.joblib"

RANDOM_STATE = 42
FEATURES = ["Provinsi", "Tahun", "Luas Panen", "Curah hujan", "Kelembapan", "Suhu rata-rata"]
TARGET = "Produksi"

df = pd.read_csv(DATA, encoding="utf-8-sig")
le = LabelEncoder()
df_enc = df.copy()
df_enc["Provinsi"] = le.fit_transform(df_enc["Provinsi"])

X, y = df_enc[FEATURES], df_enc[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

scaler = StandardScaler().fit(X_train)
X_train_s = pd.DataFrame(scaler.transform(X_train), columns=FEATURES, index=X_train.index)
X_test_s = pd.DataFrame(scaler.transform(X_test), columns=FEATURES, index=X_test.index)

# (model, pakai data scaled?)
models = {
    "Decision Tree": (DecisionTreeRegressor(random_state=RANDOM_STATE), False),
    "KNN": (KNeighborsRegressor(n_neighbors=5), True),
    "MLP": (MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=RANDOM_STATE,
                         early_stopping=True, validation_fraction=0.1), True),
    "Random Forest": (RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE), False),
    "XGBoost": (XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8,
                             random_state=RANDOM_STATE, verbosity=0), False),
    "LightGBM": (LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8,
                               random_state=RANDOM_STATE, verbose=-1), False),
}

bundle_models, metrics = {}, []
for name, (model, use_scaled) in models.items():
    Xtr, Xte = (X_train_s, X_test_s) if use_scaled else (X_train, X_test)
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    metrics.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "R2": r2_score(y_test, pred),
        "Pearson r": float(pearsonr(y_test, pred)[0]),
    })
    bundle_models[name] = {"model": model, "use_scaled": use_scaled}
    print(f"{name:14s} R2={metrics[-1]['R2']:.4f}  MAE={metrics[-1]['MAE']:,.0f}")

metrics_df = pd.DataFrame(metrics).sort_values("R2", ascending=False).reset_index(drop=True)
best = metrics_df.loc[0, "Model"]
print(f"\nModel terbaik (R2): {best}")

joblib.dump({
    "models": bundle_models,
    "scaler": scaler,
    "label_encoder": le,
    "features": FEATURES,
    "metrics": metrics_df,
    "best_model": best,
    "provinsi": list(le.classes_),
    "ranges": df[FEATURES[1:]].agg(["min", "max", "median"]).to_dict(),
}, OUT)
print(f"Tersimpan: {OUT}")
