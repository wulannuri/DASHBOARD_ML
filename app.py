import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Deteksi Ujaran Kebencian TikTok",
    page_icon="💬",
    layout="wide"
)

# =========================
# CUSTOM CSS (PINK THEME)
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #fff0f6, #ffe4ec);
}

h1, h2, h3 {
    color: #222222 !important;
    font-weight: 800;
}

.main-title {
    font-size: 44px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 10px;
    background: linear-gradient(to right, #ff5fa2, #ff8dc7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #222 !important;
    margin-bottom: 35px;
}

.card {
    padding: 28px;
    border-radius: 22px;
    background: #ffffff;
    box-shadow: 0 10px 25px rgba(255, 95, 162, 0.18);
    margin-bottom: 25px;
    color: #222 !important;
}

.card h3 {
    color: #ff4f9a !important;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ff5fa2, #ff8dc7);
}

section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

.stButton>button {
    background: linear-gradient(to right, #ff5fa2, #ff8dc7);
    color: #ffffff !important;
    border: none;
    border-radius: 14px;
    padding: 10px 22px;
    font-weight: bold;
}

textarea {
    background-color: #ffffff !important;
    border-radius: 14px !important;
    border: 2px solid #ff9acb !important;
    color: #222222 !important;
}

[data-testid="stMetric"] {
    background: #ffffff;
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 5px 15px rgba(255, 95, 162, 0.18);
}

label,
label span,
[data-testid="stMetricLabel"] {
    color: #000000 !important;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #222222 !important;
    font-size: 36px !important;
    font-weight: 800;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_tools():
    model = joblib.load("rf_tfidf_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, vectorizer, label_encoder

model, vectorizer, le = load_tools()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("📌 Menu Navigasi")
menu = st.sidebar.radio(
    "",
    [
        "🏠 Beranda",
        "📂 Dataset",
        "📊 Model & Evaluasi",
        "🔍 Prediksi Teks Baru",
        "📁 Prediksi File"
    ]
)

# =========================
# BERANDA
# =========================
if menu == "🏠 Beranda":
    st.markdown('<div class="main-title">Deteksi Ujaran Kebencian TikTok</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Klasifikasi emosi komentar TikTok menggunakan <b>Random Forest + TF-IDF</b></div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">
        <h3>🎭 Kelas Emosi</h3>
        <ul>
            <li>🔥 Benci</li>
            <li>😞 Kecewa</li>
            <li>😡 Marah</li>
            <li>😢 Sedih</li>
            <li>😊 Senang</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# =========================
# DATASET
# =========================
elif menu == "📂 Dataset":
    st.title("📂 Dataset")
    df = pd.read_csv("teks.csv")

    col1, col2 = st.columns(2)
    col1.metric("Jumlah Data", df.shape[0])
    col2.metric("Jumlah Kolom", df.shape[1])

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# MODEL & EVALUASI
# =========================
elif menu == "📊 Model & Evaluasi":
    st.title("📊 Model & Evaluasi")

    st.markdown("""
    <div class="card">
        <h3>✅ Model Terbaik</h3>
        <p><b>Random Forest + TF-IDF</b></p>
        <p>
        Model Random Forest dengan fitur TF-IDF menghasilkan performa terbaik dengan akurasi
        tertinggi sebesar <b>99%</b>. Keunggulan ini disebabkan oleh pendekatan ensemble learning
        yang mampu menangkap pola kompleks serta mengurangi overfitting. Representasi TF-IDF
        juga memberikan bobot kata yang lebih informatif dibandingkan metode lain.
        </p>
    </div>
    """, unsafe_allow_html=True)

# =========================
# PREDIKSI TEKS BARU
# =========================
elif menu == "🔍 Prediksi Teks Baru":
    st.title("🔍 Prediksi Komentar TikTok")

    st.markdown('<div class="card">', unsafe_allow_html=True)

    text_input = st.text_area(
        "✍️ Masukkan komentar:",
        height=150,
        placeholder="Contoh: Dasar tidak tahu diri!"
    )

    if st.button("🚀 Prediksi"):
        if text_input.strip() == "":
            st.warning("⚠️ Teks tidak boleh kosong.")
        else:
            vec = vectorizer.transform([text_input])
            pred = model.predict(vec)[0]
            label = le.inverse_transform([pred])[0]
            probs = model.predict_proba(vec)[0]

            st.success(f"🔍 **Emosi:** {label.upper()}")
            st.progress(float(np.max(probs)))

            fig, ax = plt.subplots()
            ax.bar(le.classes_, probs)
            ax.set_ylim(0, 1)
            plt.xticks(rotation=45)
            st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# PREDIKSI FILE
# =========================
elif menu == "📁 Prediksi File":
    st.title("📁 Prediksi dari File")

    st.markdown('<div class="card">', unsafe_allow_html=True)

    file = st.file_uploader("Upload file CSV atau TXT", type=["csv", "txt"])

    if file is not None:
        if file.name.endswith(".txt"):
            text = file.read().decode("utf-8")
            vec = vectorizer.transform([text])
            label = le.inverse_transform(model.predict(vec))[0]
            st.success(f"🔍 **Emosi:** {label.upper()}")

        else:
            df = pd.read_csv(file)
            st.dataframe(df.head(), use_container_width=True)

            col = st.selectbox("Pilih kolom teks:", df.columns)

            if st.button("🚀 Prediksi File"):
                vecs = vectorizer.transform(df[col].astype(str))
                preds = model.predict(vecs)
                df["Prediksi_Emosi"] = le.inverse_transform(preds)

                st.success("✅ Prediksi selesai")
                st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Hasil",
                    csv,
                    "hasil_prediksi.csv",
                    "text/csv"
                )

    st.markdown('</div>', unsafe_allow_html=True)
