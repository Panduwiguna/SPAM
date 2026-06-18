import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression # Sesuaikan dengan model yang Anda gunakan (misal: MultinomialNB)
from sklearn.model_selection import train_test_split

# =========================================================================
# TAHAP 1: SIMPAN PROSES TRAINING DI MEMORI (SINKRONISASI OTOMATIS)
# =========================================================================
@st.cache_resource
def train_and_get_model():
    # Pastikan file SMSSpamCollection ada di folder utama Github Anda
    # Membaca data dataset asli
    df = pd.read_txt('SMSSpamCollection', sep='\t', names=['label', 'sms'])
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    
    X = df['sms']
    y = df['label_num']
    
    # Split data dengan random_state konstan
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Inisialisasi Vectorizer dan Model
    vectorizer = CountVectorizer()
    model = LogisticRegression() # Ubah ke MultinomialNB() jika modul Anda menggunakan Naive Bayes
    
    # Proses Fit dan Transform data train
    X_train_vec = vectorizer.fit_transform(X_train)
    model.fit(X_train_vec, y_train)
    
    return model, vectorizer

# Memanggil fungsi training di server Streamlit
try:
    model, vectorizer = train_and_get_model()
except Exception as e:
    st.error(f"Gagal membaca dataset 'SMSSpamCollection'. Pastikan file tersebut sudah diunggah ke GitHub.")

# =========================================================================
# TAHAP 2: ANTARMUKA STREAMLIT (UI)
# =========================================================================
st.title("📨 SMS Spam Detection")
st.write("Cek apakah SMS termasuk spam:")

user_input = st.text_input("Masukkan pesan SMS:")

if st.button("Prediksi") or user_input:
    if user_input.strip() != "":
        # Mengubah teks input baru dengan vectorizer yang sama persis
        input_vec = vectorizer.transform([user_input])
        
        # Eksekusi Prediksi
        pred = model.predict(input_vec)[0]
        
        # Tampilkan hasil penentuan kelas
        if pred == 1:
            st.error("Hasil Prediksi: SPAM 🚨")
        else:
            st.success("Hasil Prediksi: HAM (Bukan Spam) ✅")
    else:
        st.warning("Silakan masukkan teks SMS terlebih dahulu.")
