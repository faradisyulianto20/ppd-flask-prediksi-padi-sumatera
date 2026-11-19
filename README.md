# Prediksi Produksi Padi Sumatera (Flask Dashboard)

Deployment model forecasting produksi padi (ton) berdasarkan provinsi, tahun, luas panen,
curah hujan, kelembapan, dan suhu rata-rata. Data BPS & BMKG 1993-2020, 8 provinsi di Sumatera.
Praktikum Penambangan Data - Pertemuan 10: Model Deployment (Flask).

## Isi
| File | Fungsi |
|---|---|
| `train_model.py` | Melatih 6 model (Decision Tree, KNN, MLP, Random Forest, XGBoost, LightGBM) dan menyimpan `model_padi.joblib` |
| `flask_app.py` | Dashboard Flask (`/padi/` dan `/padi/result/`) |
| `app.py` | Versi alternatif dengan Streamlit |
| `templates/` | Template HTML (Jinja2 + Bootstrap) |
| `model_padi.joblib` | Paket model, scaler, label encoder, dan metrik |
| `Data_Tanaman_Padi_Sumatera_version_1.csv` | Dataset |
| `uts_ppd_kel_5_padi_sumatera.ipynb` | Notebook eksplorasi dan pemodelan |

## Menjalankan
```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

python train_model.py           # opsional, melatih ulang model
python flask_app.py             # buka http://127.0.0.1:5001/padi/
```

Catatan: `model_padi.joblib` dibuat dengan scikit-learn 1.8.0. Jika versi berbeda dan model gagal
dimuat, jalankan ulang `python train_model.py`.

## Hasil (data uji 20%)
XGBoost terbaik dengan R2 0,9744 dan MAE sekitar 103 ribu ton. MLP tidak andal (R2 negatif).
