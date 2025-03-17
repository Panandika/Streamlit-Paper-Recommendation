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
        
        # Pastikan kategorinya berupa list (bukan string)
        raw_categories = data['categories'][idx]
        if isinstance(raw_categories, str):
            raw_categories = ast.literal_eval(raw_categories)

        # Mapping kategori
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
    'Pendahuluan': 'embeddings_intro_no_prepro',
    'Kombinasi Teks': 'embeddings_combined_no_prepro'
}

selected_embedding = st.radio(
    "Pilih Jenis Embedding yang Akan Digunakan:",
    options=list(embedding_options.keys()),
    index=3,  # Default ke Kombinasi Teks
    horizontal=True
)


# User input
input_text = st.text_area(
    "Enter the topics or keywords you are interested in:",
    placeholder="Example: deep learning for natural language processing"
    key="search_input
)

if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'show_feedback' not in st.session_state:
    st.session_state.show_feedback = False

# Recommendations
if st.button("Search Journal"):
    if not input_text.strip():
        st.error("Please enter a topic or keyword.")
    else:
        with st.spinner("Searching Journal..."):
            recommendations = get_recommendations(
                input_text, 
                model, 
                data,
                embedding_column=embedding_options[selected_embedding]
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