import streamlit as st
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import pickle
from streamlit_modal import Modal
import ast

# Mapping kategori
CATEGORY_MAPPING = {
    'cs.AI': 'Artificial Intelligence',
    'cs.CC': 'Computational Complexity',
    'cs.DB': 'Databases',
    'cs.GT': 'Game Theory',
    'cs.IR': 'Information Retrieval',
    'cs.PL': 'Programming Languages',
    'cs.SE': 'Software Engineering',
    'cs.OH': 'Other Computer Science Topics',
    'cs.RO': 'Robotics',
    'cs.CV': 'Computer Vision',
    'cs.CL': 'Computation and Language',
    'cs.LG': 'Machine Learning',
}

# Load pre-trained model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load dataset
@st.cache_resource
def load_data():
    with open('data_w_embeddings_v2.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

# Calculate cosine similarity
# Fungsi rekomendasi yang dimodifikasi
def get_recommendations(input_text, model, data, embedding_columns, top_k=5):
    # Encode input text sekali saja
    input_embedding_single = model.encode(input_text, convert_to_tensor=True)
    
    # Hitung jumlah embedding yang dipilih
    num_selected = len(embedding_columns)
    
    # Gabungkan embedding input sesuai jumlah yang dipilih
    input_embedding = torch.cat([input_embedding_single] * num_selected)
    
    # Siapkan embeddings untuk semua paper
    device = input_embedding.device
    paper_embeddings = []
    
    for idx in range(len(data)):
        # Kumpulkan semua embedding yang dipilih untuk paper ini
        combined = []
        for col in embedding_columns:
            combined.append(data[col][idx])
        
        # Gabungkan menjadi satu array numpy
        combined_np = np.concatenate(combined)
        paper_embeddings.append(combined_np)
    
    # Konversi ke tensor dan pindahkan ke device yang sama
    paper_embeddings = torch.tensor(np.array(paper_embeddings), device=device)
    
    # Hitung similarity
    cosine_scores = util.cos_sim(input_embedding.unsqueeze(0), paper_embeddings)
    
    # Ambil hasil terbaik
    top_results = torch.topk(cosine_scores, k=top_k)
    
    recommendations = []
    for idx in top_results.indices[0]:
        idx = idx.item()
        
        # Proses kategori sama seperti sebelumnya
        raw_categories = data['categories'][idx]
        if isinstance(raw_categories, str):
            raw_categories = ast.literal_eval(raw_categories)
        
        category_names = [CATEGORY_MAPPING.get(cat, cat) for cat in raw_categories]
        category_display = ' / '.join(category_names)

        recommendations.append({
            'title': data['title'][idx],
            'abstract': data['summary'][idx],
            'category': category_display,
            'pdf_url': data['pdf_url'][idx]
        })
    
    return recommendations

# Streamlit App
st.set_page_config(page_title="Journal Recommendation System", layout="centered")
st.title("Scientific Journal Recommendation System")

st.sidebar.header("About")
st.sidebar.write(
    "This application recommends scientific journals based on user input using the BERT model and cosine similarity."
)
st.sidebar.markdown("[Give Feedback](https://forms.gle/7kCtB3nvRbzhetL2A)")

# Load model and data
model = load_model()
data = load_data()

# Embedding selection
embedding_options = {
    'Judul': 'embeddings_title', 
    'Abstrak': 'embeddings_abstract_no_prepro',
    'Pendahuluan': 'embeddings_intro_no_prepro'
}

selected_embedding_types = st.multiselect(
    "Choose Similarity Options:",
    options=list(embedding_options.keys()),
    default=['Judul', 'Abstrak', 'Pendahuluan'],  # Default kombinasi lengkap
    key="embedding_selector"
)

# User input
input_text = st.text_area(
    "Enter the topics or keywords you are interested in:",
    placeholder="Example: deep learning for natural language processing",
    key="search_input"
)

if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'show_feedback' not in st.session_state:
    st.session_state.show_feedback = False

# Recommendations
if st.button("Search Journal"):
    if not input_text.strip():
        st.error("Harap masukkan topik atau kata kunci.")
    else:
        with st.spinner("Mencari Journal..."):
            # Dapatkan kolom embedding yang dipilih
            selected_columns = [embedding_options[t] for t in selected_embedding_types]
            
            recommendations = get_recommendations(
                input_text, 
                model, 
                data,
                embedding_columns=selected_columns  # Kirim list kolom yang dipilih
            )
        
        st.session_state.search_results = recommendations
        st.session_state.show_feedback = True


# Persistent results display
if st.session_state.search_results:
    st.subheader("Journal Recommendations:")
    for rec in st.session_state.search_results:
        st.markdown(f"### [{rec['title']}]({rec['pdf_url']})")
        st.markdown(f"**Category**: {rec['category']}")
        st.markdown(f"**Abstract**: {rec['abstract']}")
        st.write("---")

feedback_modal = Modal(key="feedback_modal", title="Give Feedback 🗣️")

if st.session_state.show_feedback:
    with feedback_modal.container():
        st.markdown("Help us improve this recommendation system!")
        st.markdown("[Click here to fill out the feedback form](https://forms.gle/7kCtB3nvRbzhetL2A)")
    
    # Reset feedback trigger without clearing results
    st.session_state.show_feedback = False