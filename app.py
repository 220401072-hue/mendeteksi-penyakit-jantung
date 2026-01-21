import os
from flask import Flask, request, jsonify
import pandas as pd
import tensorflow as tf
import joblib
import pickle
import numpy as np

# Inisialisasi Aplikasi Flask
app = Flask(__name__)

# =========================================================================
# BAGIAN INI YANG MEMPERBAIKI ERROR "FILE NOT FOUND"
# =========================================================================
# Mendapatkan lokasi folder tempat file api.py ini berada
base_dir = os.path.dirname(os.path.abspath(__file__))

print(f"📂 Folder Proyek terdeteksi di: {base_dir}")

# Menggabungkan path folder dengan nama file agar selalu akurat
model_path = os.path.join(base_dir, 'heart_disease_model.h5')
scaler_path = os.path.join(base_dir, 'scaler.pkl')
columns_path = os.path.join(base_dir, 'model_columns.pkl')

# =========================================================================
# 1. LOAD MODEL & ASSETS
# =========================================================================
print("⏳ Sedang memuat model Deep Learning...")

try:
    # Cek apakah file benar-benar ada sebelum di-load
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File model tidak ditemukan di: {model_path}")

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    with open(columns_path, 'rb') as f:
        model_columns = pickle.load(f)
        
    print("✅ SUKSES: Model siap digunakan!")

except Exception as e:
    print("\n❌ ERROR FATAL saat memuat model:")
    print(e)
    print("Pastikan file .h5, .pkl ada di folder yang sama dengan api.py!")
    exit() # Berhenti jika model gagal load

# =========================================================================
# 2. ENDPOINT PREDIKSI
# =========================================================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Terima data JSON dari PHP
        json_data = request.json
        
        # Buat DataFrame
        input_df = pd.DataFrame([json_data])
        
        # Konversi string "true"/"false" dari PHP menjadi boolean Python
        bool_cols = ['fbs', 'exang']
        for col in bool_cols:
            if col in input_df.columns:
                input_df[col] = input_df[col].apply(lambda x: True if str(x).lower() == 'true' else False)

        # One-Hot Encoding
        df_processed = pd.get_dummies(input_df)
        
        # Penyesuaian Kolom (Reindexing) agar sesuai format Training
        df_final = df_processed.reindex(columns=model_columns, fill_value=0)
        
        # Scaling Data
        X_scaled = scaler.transform(df_final)
        
        # Prediksi Neural Network
        prediction_prob = model.predict(X_scaled)[0][0]
        prediction_prob = float(prediction_prob) # Ubah ke float biasa
        
        # Logika Hasil
        result = {
            "prediction": 1 if prediction_prob > 0.5 else 0,
            "confidence": prediction_prob if prediction_prob > 0.5 else (1 - prediction_prob),
            "status": "success"
        }
        
        return jsonify(result)

    except Exception as e:
        # Kirim pesan error ke PHP jika ada masalah
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    # Menjalankan server di Port 5000
    app.run(port=5000, debug=True)
