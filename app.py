import streamlit as st
import pandas as pd
import tensorflow as tf
import joblib
import pickle
import numpy as np
import os

# Konfigurasi Halaman
st.set_page_config(
    page_title="Prediksi Kesehatan Jantung",
    page_icon="🫀",
    layout="centered"
)

# Judul dan Deskripsi
st.title("🫀 Smart Heart Disease Prediction")
st.info("Masukkan data klinis pasien di bawah ini untuk mendapatkan hasil prediksi.")

# =========================================================================
# 1. LOAD MODEL & ASSETS (Menggunakan Cache agar cepat)
# =========================================================================
@st.cache_resource
def load_assets():
    try:
        # Load Model H5
        model = tf.keras.models.load_model('heart_disease_model.h5')
        
        # Load Scaler
        scaler = joblib.load('scaler.pkl')
        
        # Load Columns (untuk urutan fitur one-hot encoding)
        with open('model_columns.pkl', 'rb') as f:
            model_columns = pickle.load(f)
            
        return model, scaler, model_columns
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None

model, scaler, model_columns = load_assets()

# =========================================================================
# 2. FORM INPUT DATA (Pengganti index.php)
# =========================================================================
if model and scaler and model_columns:
    with st.form("prediction_form"):
        st.subheader("Data Pasien")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Usia (Age)", min_value=1, max_value=120, value=50)
            sex = st.selectbox("Jenis Kelamin", ["Male", "Female"])
            cp = st.selectbox("Tipe Nyeri Dada (Chest Pain)", 
                              ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
            trestbps = st.number_input("Tekanan Darah (mm Hg)", min_value=50, max_value=250, value=120)
            chol = st.number_input("Kolesterol (mg/dl)", min_value=100, max_value=600, value=200)
            fbs = st.checkbox("Gula Darah Puasa > 120 mg/dl?")
            
        with col2:
            restecg = st.selectbox("Hasil EKG Istirahat", 
                                   ["normal", "lv hypertrophy", "st-t abnormality"])
            thalch = st.number_input("Detak Jantung Maksimum", min_value=50, max_value=250, value=150)
            exang = st.checkbox("Nyeri Dada Akibat Olahraga (Exang)?")
            oldpeak = st.number_input("ST Depression (Oldpeak)", value=0.0, format="%.1f")
            slope = st.selectbox("Slope ST Segment", ["upsloping", "flat", "downsloping"])
            ca = st.number_input("Jumlah Pembuluh Utama (0-3)", min_value=0, max_value=3, value=0)
            thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])

        submit_btn = st.form_submit_button("Prediksi Sekarang")

    # =========================================================================
    # 3. LOGIKA PREDIKSI (Diadaptasi dari api.py)
    # =========================================================================
    if submit_btn:
        try:
            # 1. Siapkan data mentah dictionary
            input_data = {
                'age': age,
                'sex': sex,
                'dataset': 'Cleveland', # Default value as placeholder if needed
                'cp': cp,
                'trestbps': trestbps,
                'chol': chol,
                'fbs': fbs, # Streamlit checkbox returns boolean True/False directly
                'restecg': restecg,
                'thalch': thalch,
                'exang': exang,
                'oldpeak': oldpeak,
                'slope': slope,
                'ca': str(ca), # Ensure consistent type
                'thal': thal
            }
            
            # 2. Buat DataFrame
            input_df = pd.DataFrame([input_data])
            
            # 3. One-Hot Encoding
            df_processed = pd.get_dummies(input_df)
            
            # 4. Reindexing (PENTING: Agar kolom sesuai dengan saat training)
            # Isi kolom yang hilang dengan 0
            df_final = df_processed.reindex(columns=model_columns, fill_value=0)
            
            # 5. Scaling
            X_scaled = scaler.transform(df_final)
            
            # 6. Prediksi
            prediction_prob = model.predict(X_scaled)[0][0]
            prediction_prob = float(prediction_prob)
            
            # 7. Tampilkan Hasil
            st.markdown("---")
            if prediction_prob > 0.5:
                st.error(f"⚠️ **Risiko Tinggi Terdeteksi**")
                st.write(f"Probabilitas Penyakit Jantung: **{prediction_prob*100:.2f}%**")
                st.warning("Saran: Segera konsultasikan dengan dokter kardiologi.")
            else:
                st.success(f"✅ **Kondisi Jantung Tampak Sehat**")
                st.write(f"Probabilitas Penyakit Jantung: **{prediction_prob*100:.2f}%**")
                st.info("Saran: Tetap jaga pola hidup sehat dan olahraga teratur.")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses data: {e}")


