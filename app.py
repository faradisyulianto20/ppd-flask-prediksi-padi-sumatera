"""Dashboard Streamlit - Forecasting Produksi Padi Sumatera.

Jalankan:  python -m streamlit run app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

HERE = Path(__file__).parent
MODEL_PATH = HERE / "model_padi.joblib"
DATA_PATH = HERE / "Data_Tanaman_Padi_Sumatera_version_1.csv"

st.set_page_config(page_title="Prediksi Produksi Padi Sumatera", page_icon="🌾", layout="wide")


@st.cache_resource(show_spinner=True)
def load_bundle(path: Path = MODEL_PATH):
    return joblib.load(path)


@st.cache_data
def load_data(path: Path = DATA_PATH):
    return pd.read_csv(path, encoding="utf-8-sig")


if not MODEL_PATH.exists():
    st.error("File model_padi.joblib belum ada. Jalankan dulu: python train_model.py")
    st.stop()

bundle = load_bundle()
df = load_data()
rng = bundle["ranges"]

st.title("🌾 Prediksi Produksi Padi di Sumatera")
st.caption("Forecasting produksi padi (ton) berdasarkan luas panen dan data iklim · Data BPS & BMKG 1993–2020")

tab_pred, tab_eval, tab_data = st.tabs(["Prediksi", "Evaluasi Model", "Eksplorasi Data"])

# ---------------------------------------------------------------- Prediksi
with tab_pred:
    model_name = st.sidebar.selectbox(
        "Model", list(bundle["models"]),
        index=list(bundle["models"]).index(bundle["best_model"]),
        help="Default: model dengan R² tertinggi",
    )
    st.sidebar.caption(f"Model terbaik: **{bundle['best_model']}**")

    with st.form("input-form"):
        st.subheader("Masukkan data")
        c1, c2 = st.columns(2)
        with c1:
            provinsi = st.selectbox("Provinsi", bundle["provinsi"])
            tahun = st.number_input("Tahun", min_value=1990, max_value=2050, value=2021, step=1)
            luas = st.number_input("Luas Panen (ha)", min_value=0.0, value=float(rng["Luas Panen"]["median"]), step=1000.0)
        with c2:
            hujan = st.number_input("Curah hujan (mm)", min_value=0.0, value=float(rng["Curah hujan"]["median"]), step=10.0)
            lembap = st.number_input("Kelembapan (%)", min_value=0.0, max_value=100.0, value=float(rng["Kelembapan"]["median"]), step=0.1)
            suhu = st.number_input("Suhu rata-rata (°C)", min_value=0.0, max_value=50.0, value=float(rng["Suhu rata-rata"]["median"]), step=0.1)
        submitted = st.form_submit_button("Prediksi", type="primary")

    if submitted:
        cfg = bundle["models"][model_name]
        row = pd.DataFrame([[
            bundle["label_encoder"].transform([provinsi])[0], tahun, luas, hujan, lembap, suhu,
        ]], columns=bundle["features"])
        if cfg["use_scaled"]:
            row = pd.DataFrame(bundle["scaler"].transform(row), columns=bundle["features"])
        pred = float(cfg["model"].predict(row)[0])

        st.success("Prediksi selesai")
        st.metric(f"Prediksi produksi padi – {provinsi}, {int(tahun)} ({model_name})", f"{pred:,.0f} ton")

        hist = df[df["Provinsi"] == provinsi].sort_values("Tahun")
        fig = px.line(hist, x="Tahun", y="Produksi", markers=True, title=f"Produksi historis {provinsi} vs prediksi")
        fig.add_scatter(x=[tahun], y=[pred], mode="markers", name="Prediksi",
                        marker=dict(size=14, color="red", symbol="star"))
        st.plotly_chart(fig, width='stretch')

        if model_name == "MLP":
            st.warning("MLP memiliki R² negatif pada data uji, hasilnya tidak dapat diandalkan.")

# ---------------------------------------------------------------- Evaluasi
with tab_eval:
    st.subheader("Perbandingan 6 model (data uji 20%)")
    m = bundle["metrics"]
    st.dataframe(
        m.style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "R2": "{:.4f}", "Pearson r": "{:.4f}"}),
        hide_index=True, width='stretch',
    )
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(m, x="Model", y="R2", title="R²"), width='stretch')
    c2.plotly_chart(px.bar(m, x="Model", y="MAE", title="MAE (ton)"), width='stretch')

# ---------------------------------------------------------------- Data
with tab_data:
    st.subheader("Tren produksi per provinsi")
    pilih = st.multiselect("Provinsi", bundle["provinsi"], default=bundle["provinsi"])
    sub = df[df["Provinsi"].isin(pilih)]
    st.plotly_chart(px.line(sub, x="Tahun", y="Produksi", color="Provinsi", markers=True), width='stretch')
    with st.expander("Tabel data"):
        st.dataframe(sub, hide_index=True, width='stretch')
