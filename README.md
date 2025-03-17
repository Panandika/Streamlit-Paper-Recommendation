# 💻 Streamlit Scientific Paper Recommendation

A Streamlit-based web application that recommends scientific papers based on user-provided keywords or topics. This app leverages Sentence-BERT (SBERT) embeddings and cosine similarity to suggest the most relevant journals from a preprocessed dataset of arXiv papers.

## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://paper-recommendation.streamlit.app)


## 🔍 How It Works

This application uses natural language processing (NLP) techniques to provide personalized journal recommendations:

Sentence-BERT (all-MiniLM-L6-v2) is used to convert the input text and each paper into vector embeddings.
The embeddings represent the semantic meaning of text and are compared using cosine similarity.
The app then returns the top-k most relevant papers based on the similarity scores.

## Key Features:
🔎 Multi-embedding options: Users can choose whether to match their input with the title, abstract, introduction, or combined text of papers.
📚 Category mapping: Each journal is tagged with a simplified category name based on arXiv classifications.
📈 Fast & efficient: Uses caching for both model and dataset loading, improving user experience and performance.
💬 Feedback integration: Users can easily provide feedback to improve the system.

## ⚙️ Technologies Used
Streamlit – for building interactive web interfaces
Sentence-Transformers – for generating text embeddings
PyTorch – backend support for model inference
Pickle – to load precomputed data
[Pandas, NumPy] – for data handling
streamlit-modal – feedback modal implementation

## 📈 Example Output
Users input a research topic, such as:

"deep learning for natural language processing"

The system returns the top 5 most semantically relevant scientific papers with:

Title (clickable link to PDF)
Abstract
Simplified Categories

