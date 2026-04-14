import streamlit as st
import pandas as pd
import joblib
import numpy as np
from numpy.linalg import norm

# 1. Load Data dan Model (Gunakan cache agar loading website lebih cepat)
@st.cache_resource
def load_data_and_model():
    # Pastikan file anime_reference.csv yang ini SUDAH ADA kolom 'genre'-nya
    anime_df = pd.read_csv('anime_reference.csv')
    model = joblib.load('svd_model.pkl')
    return anime_df, model

anime_df, model = load_data_and_model()

# Mengurutkan judul anime sesuai abjad
anime_titles = anime_df['title'].dropna().unique().tolist()
anime_titles.sort()

# 2. Antarmuka Website
st.title("🎬 Sistem Rekomendasi Anime")
st.write("Temukan anime baru berdasarkan algoritma SVD by Fhariel!")

# Input user
selected_anime = st.selectbox("Pilih Anime yang kamu suka:", anime_titles)

if st.button("Dapatkan Rekomendasi", type="primary"):
    st.write(f"Mencari rekomendasi untuk **{selected_anime}**...")
    
    try:
        # 3. Logika Rekomendasi SVD
        anime_raw_id = anime_df[anime_df['title'] == selected_anime]['anime_id'].values[0]
        anime_inner_id = model.trainset.to_inner_iid(anime_raw_id)
        
        item_factors = model.qi 
        target_factor = item_factors[anime_inner_id]
        
        # Hitung kemiripan (Cosine Similarity)
        similarities = np.dot(item_factors, target_factor) / (norm(item_factors, axis=1) * norm(target_factor))
        
        # Ambil top 10 teratas (index 1 sampai 11, karena index 0 adalah anime itu sendiri)
        similar_indices = similarities.argsort()[::-1][1:11]
        
        st.subheader(f"Top 10 mirip dengan {selected_anime}:")
        
        # 4. Tampilkan Hasil menggunakan Custom HTML + CSS
        for idx in similar_indices:
            raw_iid = model.trainset.to_raw_iid(idx)
            
            # Ambil data judul dan genre
            recom_data = anime_df[anime_df['anime_id'] == raw_iid].iloc[0]
            recom_title = recom_data['title']
            recom_genre = recom_data['genre'] # Jika di Colab namamu 'genres', ubah kata 'genre' ini
            
            # --- INI ADALAH BAGIAN HTML & CSS NYA ---
            html_card = f"""
            <div style="
                background-color: #2b303a; 
                padding: 15px; 
                border-radius: 10px; 
                margin-bottom: 12px; 
                border-left: 6px solid #ff4b4b;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            ">
                <h4 style="margin: 0; color: #ffffff; font-family: sans-serif;">{recom_title}</h4>
                <p style="margin: 6px 0 0 0; color: #b2b2b2; font-size: 14px; font-family: sans-serif;">
                    🏷️ <b>Genre:</b> {recom_genre}
                </p>
            </div>
            """
            
            # Menyuntikkan HTML ke Streamlit
            st.markdown(html_card, unsafe_allow_html=True)
            
    except Exception as e:
        st.error("Maaf, tidak bisa memproses anime ini atau data kurang lengkap.")
