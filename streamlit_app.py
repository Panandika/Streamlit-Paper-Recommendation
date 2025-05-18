import streamlit as st
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import pickle
from streamlit_modal import Modal
import ast
import datetime

# Set page config at the very beginning
st.set_page_config(page_title="Journal Recommendation System", layout="centered")

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
'''

'''
# Load dataset
@st.cache_resource
def load_data():
    with open('data_w_embeddings_v4_with_url.pkl', 'rb') as file:
        data = pickle.load(file)
    
    # Convert timestamps to years for filtering
    data['year'] = data['published'].apply(
        lambda x: datetime.datetime.fromtimestamp(int(x)/1000).year if pd.notna(x) else None
    )
    return data

# Calculate cosine similarity
# Fungsi rekomendasi yang dimodifikasi
def get_recommendations(input_text, model, data, embedding_columns, year_range=None, top_k=5):
    # Filter data by year if year_range is provided
    filtered_data = data
    if year_range:
        min_year, max_year = year_range
        filtered_data = data[
            (data['year'] >= min_year) & 
            (data['year'] <= max_year)
        ]
        
        # If no papers match the year filter, return empty recommendations
        if len(filtered_data) == 0:
            return []
    
    # Encode input text sekali saja
    input_embedding_single = model.encode(input_text, convert_to_tensor=True)
    
    # Hitung jumlah embedding yang dipilih
    num_selected = len(embedding_columns)
    
    # Gabungkan embedding input sesuai jumlah yang dipilih
    input_embedding = torch.cat([input_embedding_single] * num_selected)
    
    # Siapkan embeddings untuk semua paper
    device = input_embedding.device
    paper_embeddings = []
    
    for idx in range(len(filtered_data)):
        # Kumpulkan semua embedding yang dipilih untuk paper ini
        combined = []
        for col in embedding_columns:
            combined.append(filtered_data[col].iloc[idx])
        
        # Gabungkan menjadi satu array numpy
        combined_np = np.concatenate(combined)
        paper_embeddings.append(combined_np)
    
    # Konversi ke tensor dan pindahkan ke device yang sama
    paper_embeddings = torch.tensor(np.array(paper_embeddings), device=device)
    
    # Hitung similarity
    cosine_scores = util.cos_sim(input_embedding.unsqueeze(0), paper_embeddings)
    
    # Ambil hasil terbaik
    top_results = torch.topk(cosine_scores, k=min(top_k, len(filtered_data)))
    
    recommendations = []
    for idx in top_results.indices[0]:
        idx = idx.item()
        
        # Proses kategori sama seperti sebelumnya
        raw_categories = filtered_data['categories'].iloc[idx]
        if isinstance(raw_categories, str):
            raw_categories = ast.literal_eval(raw_categories)
        
        category_names = [CATEGORY_MAPPING.get(cat, cat) for cat in raw_categories]
        category_display = ' / '.join(category_names)
        
        # Get publication year
        pub_year = filtered_data['year'].iloc[idx]
        year_display = f" ({pub_year})" if pub_year else ""

        recommendations.append({
            'title': filtered_data['title'].iloc[idx],
            'abstract': filtered_data['summary_no_prepro'].iloc[idx],
            'category': category_display,
            'pdf_url': filtered_data['pdf_url'].iloc[idx],
            'year': pub_year
        })
    
    return recommendations

# Streamlit App
st.title("Scientific Journal Recommendation System")

st.sidebar.header("About")
st.sidebar.write(
    "This application recommends scientific journals based on user input using the BERT model and cosine similarity."
)
st.sidebar.markdown("[Give Feedback](https://forms.gle/7kCtB3nvRbzhetL2A)")

# Load model and data
model = load_model()
data = load_data()

# Year range for filtering
min_year = int(data['year'].min())
max_year = int(data['year'].max())

# Add year filter in sidebar
st.sidebar.header("Filter Options")
use_year_filter = st.sidebar.checkbox("Filter by Year", value=False)

if use_year_filter:
    year_range = st.sidebar.slider(
        "Publication Year Range:",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
else:
    year_range = None

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
                embedding_columns=selected_columns,
                year_range=year_range
            )
        
        if not recommendations:
            st.warning("No journals found matching your criteria. Try adjusting your filters.")
        else:
            st.session_state.search_results = recommendations
            st.session_state.show_feedback = True


# Persistent results display
if st.session_state.search_results:
    st.subheader("Journal Recommendations:")
    for rec in st.session_state.search_results:
        year_display = f" ({rec['year']})" if rec['year'] else ""
        st.markdown(f"### [{rec['title']}{year_display}]({rec['pdf_url']})")
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