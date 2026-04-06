import streamlit as st
import pandas as pd
import joblib

# 1. Load Data dan Model
@st.cache_resource
def load_data_and_model():
    # Load data anime
    anime_df = pd.read_csv('anime_reference.csv')
    # Load model SVD
    model = joblib.load('svd_model.pkl')
    return anime_df, model

anime_df, model = load_data_and_model()

# 2. Desain Antarmuka Website
st.title("🎬 Sistem Rekomendasi Anime")
st.write("Dibuat menggunakan algoritma SVD (Matrix Factorization) berdasarkan rating pengguna MyAnimeList.")

# Membuat dropdown untuk memilih judul anime
anime_titles = anime_df['title'].dropna().unique().tolist()
selected_anime = st.selectbox("Pilih Anime yang kamu suka:", anime_titles)

if st.button("Dapatkan Rekomendasi"):
    st.write(f"Mencari rekomendasi berdasarkan kemiripan dengan **{selected_anime}**...")
    
    # 3. Logika Rekomendasi 
    # (Catatan: Karena SVD merekomendasikan berdasarkan User-Item, 
    # untuk Item-to-Item biasanya kita mengambil nilai embedding/faktor dari model)
    try:
        # Mengambil internal ID dari SVD model
        anime_raw_id = anime_df[anime_df['title'] == selected_anime]['anime_id'].values[0]
        anime_inner_id = model.trainset.to_inner_iid(anime_raw_id)
        
        # Mendapatkan faktor laten (embeddings) dari item
        item_factors = model.qi 
        
        # Menghitung similarity (misal dengan Cosine Similarity)
        import numpy as np
        from numpy.linalg import norm
        
        target_factor = item_factors[anime_inner_id]
        similarities = np.dot(item_factors, target_factor) / (norm(item_factors, axis=1) * norm(target_factor))
        
        # Mengambil top 10 index dengan nilai similarity tertinggi (mengabaikan item itu sendiri)
        similar_indices = similarities.argsort()[::-1][1:11]
        
        # Mengonversi kembali ke raw ID dan mendapatkan judul
        st.subheader("Top 10 Rekomendasi untukmu:")
        for idx in similar_indices:
            raw_iid = model.trainset.to_raw_iid(idx)
            recom_title = anime_df[anime_df['anime_id'] == raw_iid]['title'].values[0]
            st.success(f"- {recom_title}")
            
    except Exception as e:
        st.error(f"Maaf, anime belum cukup memiliki data rating untuk dibuat rekomendasinya. Detail error: {e}")