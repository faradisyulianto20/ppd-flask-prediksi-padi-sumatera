"""Dashboard Flask - Forecasting Produksi Padi Sumatera.

Jalankan:  python flask_app.py   ->  http://127.0.0.1:5001/padi/
"""
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

HERE = Path(__file__).parent
bundle = joblib.load(HERE / "model_padi.joblib")
rng = bundle["ranges"]

app = Flask(__name__)


def _num(name, default):
    try:
        return float(request.form.get(name))
    except (TypeError, ValueError):
        return default


@app.route("/padi/")
def index():
    return render_template(
        "index.html", provinsi=bundle["provinsi"], models=list(bundle["models"]),
        best=bundle["best_model"], rng=rng, form={}, metrics=None,
    )


@app.route("/padi/result/", methods=["POST"])
def result():
    f = request.form
    provinsi = f.get("provinsi")
    model_name = f.get("model", bundle["best_model"])
    if provinsi not in bundle["provinsi"] or model_name not in bundle["models"]:
        return "Provinsi atau model tidak valid", 400
    try:
        vals = [float(f[k]) for k in ("tahun", "luas", "hujan", "lembap", "suhu")]
    except (KeyError, ValueError):
        return "Semua input harus berupa angka", 400

    cfg = bundle["models"][model_name]
    row = pd.DataFrame(
        [[bundle["label_encoder"].transform([provinsi])[0], *vals]], columns=bundle["features"]
    )
    if cfg["use_scaled"]:
        row = pd.DataFrame(bundle["scaler"].transform(row), columns=bundle["features"])
    pred = float(cfg["model"].predict(row)[0])

    return render_template(
        "result.html", provinsi=provinsi, model_name=model_name, vals=vals, pred=pred,
        metrics=bundle["metrics"].to_dict("records"),
    )


if __name__ == "__main__":
    app.run(port=5001)
