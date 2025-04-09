from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from transformers import pipeline
import torch 
import gdown
import numpy as np
import pandas as pd
import dask.dataframe as dd
import pickle 
import streamlit as st 

your_model_dir = "Resources/finetunedmodel"

# Load model and tokenizer from a local folder containing safetensors

@st.cache_data
def load_file():
    amazonhkdatasetfileid = '14GcJAzyN2PFg2JuyzF0pRmxlMmimrz9o'
    # amazonhkdatasetfilename = 'AmazonHomeKitchenReviews.csv'
    amazonhkdatasetfilename = 'Resources/AmazonHomeKitchenReviews.csv'

    # url = f"https://drive.google.com/uc?export=download&id={amazonhkdatasetfileid}"
    # gdown.download(url,amazonhkdatasetfilename, quiet=False)
    df_data = dd.read_csv(amazonhkdatasetfilename)
    df_filtered = df_data[df_data['categories'] == "['Home & Kitchen', 'Bedding', 'Sheets & Pillowcases', 'Sheet & Pillowcase Sets']"]
    df_pandas = df_filtered.compute()
    df_renamed = df_pandas.rename(columns={'title_y' : 'product_title','title_x':'review_title','text':'review_text'})
    df_renamed['combined']  = df_renamed['review_title'].fillna('') + ". " + df_renamed["review_text"].fillna('') +" " +  df_renamed['product_title']
    df_renamed.dropna(subset='review_text')
    return df_renamed 

@st.cache_resource
def load_model():
    model = AutoModelForQuestionAnswering.from_pretrained(your_model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(your_model_dir)
    return pipeline("question-answering", model=model, tokenizer=tokenizer)

df_data = load_file()


qa_pipeline = load_model()

def get_context(query):
    matching_products = df_data[df_data['product_title'].str.contains(query, case=False, na=False)]
    unique_matching_df = matching_products.drop_duplicates(subset='product_title', keep='first')
    if len(unique_matching_df) == 0:
        return "No relevant reviews for this product"
        # st.markdown("No relevant reviews for this product")
        return None
    
    elif len(unique_matching_df) == 1:
        selectedtitle = unique_matching_df['product_title'].unique()
        print(selectedtitle)   
        selected_context = unique_matching_df[unique_matching_df['product_title'] == selectedtitle,'review_text']  
        selected_filter = selected_context['review_text']  
    elif  len(unique_matching_df) > 1:
        selectedtitlelist = unique_matching_df['product_title'].to_list()
        st.markdown("Multiple matching products found:")
        
        for i, title in enumerate(selectedtitlelist):
            st.markdown((f"{i + 1}. {title}"))
            # print((f"{i + 1}. {title}"))

        # choice = int(input("Select the number of the product you are interested in: "))
        choice = st.selectbox("Select product", range(1, len(selectedtitlelist)+1 ), format_func=lambda x: selectedtitlelist[x-1])
        selectedtitle = selectedtitlelist[choice - 1]
        st.write(f"You selected: {selectedtitle}")
        selected_context = unique_matching_df[unique_matching_df['product_title'].str.contains(selectedtitle, case=False, na=False)]   
        selected_filter = " ".join(selected_context['review_text'])  
    return selected_filter 


# get_context('Egyptian Cotton Sheet Set')


# --- Function to Get Answer ---
def get_distilbert_answer(query: str) -> str:
    """
    Use the fine-tuned DistilBERT model to answer the question based on the context.
    """
    context = get_context(query)
    # print(context)
    result = qa_pipeline(question=query, context=context)
    # print(result["score"])
    if result:
        # print(result["answer"])
        st.markdown(f"This is what customers are saying: {result['answer']}")
        return result['answer']
    else:
        return "Sorry, I couldn't find a answer based on the context."
    

# get_distilbert_answer('Egyptian Cotton Sheet Set')