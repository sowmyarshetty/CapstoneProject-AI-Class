
import re
import os
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
import torch
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from langchain_community.vectorstores import FAISS


# Load your fine-tuned model once

print(os.listdir("Resources/amazon-qa-model-MultipleNegativesRankingLoss"))#great it works
model_path = os.path.join("Resources/amazon-qa-model-MultipleNegativesRankingLoss/amazon-qa-model-MultipleNegativesRankingLoss")

data_path = os.path.join("Resources/review_qa_df.csv")
faiss_path = os.path.join("Resources/vectorst")
st_faiss_indexname = os.path.join("Resources/vectorst/index.faiss")



@st.cache_data
def load_data(data_path):
    data_path = os.path.join(data_path)
    review_qa_df = pd.read_csv(data_path)
    return review_qa_df

@st.cache_resource
def load_model(model_path):
    model_path = os.path.join(model_path)
    model = SentenceTransformer(model_path)
    return model

@st.cache_resource
def load_faiss_index(st_faiss_indexname):
    fais_index_st = faiss.read_index(st_faiss_indexname)
    return fais_index_st

# load_model = load_model(model_path)
model = SentenceTransformer(model_path)

#- This code  generates embeddings - will run it only one time
# - answer_texts: list of enriched answer strings
# - answer_metadata: list of metadata dicts (same order as answer_texts)
answer_texts = []
answer_metadata = []

# Load your full Q&A dataset, # Make sure this includes product_name, question, answer
datapath = os.path.join("Resources/review_qa_df.csv")
review_qa_df = load_data(datapath)

for _, row in review_qa_df.iterrows():
    product_name = row['product_title']
    rating = row['rating']
    answer = f"({product_name}, {rating}): {row['answer']}"
    answer_texts.append(answer)
    answer_metadata.append({
        'product_id': row['product_id'],
        'question': row['question'],
        'rating': rating,
        'product_name': product_name
    })

# Generate embeddings for all answers
# answer_embeddings = model.encode(answer_texts, convert_to_tensor=False, show_progress_bar=True)

# Use FAISS for faster similarity search with large datasets
# dimension = answer_embeddings.shape[1]

# Inner product (cosine on normalized vectors)
# faiss_index_st = faiss.IndexFlatIP(dimension) 

# Normalize vectors for cosine similarity
# faiss.normalize_L2(answer_embeddings) 
# faiss_index_st.add(answer_embeddings)

# if not os.path.exists(faiss_path):
#     faiss.write_index(faiss_index_st, faiss_path)
#     print(f"FAISS index saved to {faiss_path}")
# else:
#     print(f"File {faiss_path} already exists. Skipping save.")


faiss_index_st = load_faiss_index(st_faiss_indexname)

def sentencetransformerprocess_user_query(user_query, top_k=1):
    """
    Handles both metadata-style queries and normal similarity search.
    """
    user_query = user_query.lower().strip()

    # ---------------------------
    # 1. Handle rating questions
    # ---------------------------

    # Example: "what is the rating of Logitech mouse?"
    if "rating" in user_query and "what" in user_query:
        # Try to extract product name
        for meta in answer_metadata:
            if meta["product_name"].lower() in user_query:
                product = meta["product_name"]
                rating = meta["rating"]
                return f"The rating for **{product}** is **{rating} stars**."
        return "Sorry, I couldn't find the product you're asking about."

    # Example: "only show answers for products above 4 stars"
    rating_filter = None
    match = re.search(r'above (\d(\.\d)?) stars?', user_query)
    if match:
        rating_filter = float(match.group(1))

    # ------------------------------------
    # 2. Embed and normalize user question
    # ------------------------------------
    user_embedding = model.encode([user_query], convert_to_numpy=True)
    faiss.normalize_L2(user_embedding)

    # ------------------------------------
    # 3. Search with FAISS
    # ------------------------------------
    D, I = faiss_index_st.search(user_embedding, top_k)

    results = []
    for idx in I[0]:
        meta = answer_metadata[idx]

        # Apply optional rating filter
        if rating_filter is not None and meta["rating"] < rating_filter:
            continue

        product = meta["product_name"]
        rating = meta["rating"]
        answer = answer_texts[idx]

        result = f"**Product:** {product}\n**Rating:** {rating}\n**Answer:** {answer}"
        results.append(result)

    if not results:
        return "I didn't find any answers that meet your rating criteria."

    return "\n\n---\n\n".join(results)

  

