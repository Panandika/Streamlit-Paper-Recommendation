import streamlit as st
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import pickle

# Load pre-trained model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load dataset
@st.cache_resource
def load_data():
    with open('data_w_embeddings.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

# Calculate cosine similarity
def get_recommendations(input_text, model, data, embedding_column, top_k=5):
    # Encode input text to get its embedding
    input_embedding = model.encode(input_text, convert_to_tensor=True)
    
    # Ensure both tensors are on the same device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_embedding = input_embedding.to(device)
    embeddings = torch.tensor(data[embedding_column], device=device)

    # Calculate cosine similarity
    cosine_scores = util.cos_sim(input_embedding, embeddings)

    # Get top_k results
    top_results = torch.topk(cosine_scores, k=top_k)

    recommendations = []
    for idx in top_results.indices[0]:
        idx = idx.item()  # Convert tensor to integer
        recommendations.append({
            'title': data['title'][idx],
            'abstract': data['summary_no_prepro'][idx],
            'category': data['categories'][idx]
        })

    return recommendations

# Streamlit App
st.set_page_config(page_title="Sistem Rekomendasi Jurnal", layout="centered")
st.title("Sistem Rekomendasi Jurnal Ilmiah")

st.sidebar.header("Tentang Aplikasi")
st.sidebar.write(
    "Aplikasi ini merekomendasikan jurnal ilmiah berdasarkan input pengguna menggunakan model BERT dan cosine similarity."
)
st.sidebar.markdown("[Berikan Masukan](https://forms.gle/7kCtB3nvRbzhetL2A)")

# Load model and data
model = load_model()
data = load_data()

# Embedding selection
embedding_options = {
    'Abstrak': 'embeddings_abstract_no_prepro',
    'Pendahuluan': 'embeddings_intro_no_prepro',
    'Kombinasi Teks': 'embeddings_combined_no_prepro'
}

selected_embedding = st.radio(
    "Pilih Jenis Embedding yang Akan Digunakan:",
    options=list(embedding_options.keys()),
    index=2,  # Default ke Kombinasi Teks
    horizontal=True
)

# User input
input_text = st.text_area(
    "Masukkan topik atau kata kunci yang Anda minati:",
    placeholder="Contoh: deep learning for natural language processing"
)

# Recommendations
if st.button("Cari Jurnal"):
    if not input_text.strip():
        st.error("Harap masukkan topik atau kata kunci.")
    else:
        with st.spinner("Mencari jurnal..."):
            recommendations = get_recommendations(
                input_text, 
                model, 
                data,
                embedding_column=embedding_options[selected_embedding]
            )

        st.subheader("Rekomendasi Jurnal:")
        for rec in recommendations:
            st.markdown(f"### {rec['title']}")
            st.markdown(f"**Kategori**: {rec['category']}")
            st.markdown(f"**Abstrak**: {rec['abstract']}")
            st.write("---")

st.sidebar.write("Dibuat dengan Streamlit dan SentenceTransformer.")