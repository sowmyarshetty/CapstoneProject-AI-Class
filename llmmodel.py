import pandas as pd
import json
from pprint import pprint
import os
import json
import streamlit as st 
# from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from IPython.display import display, Markdown
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import TextLoader, CSVLoader, JSONLoader, PyPDFLoader
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate


# --- Paths for local execution ---
faiss_index_path = "Resources/vector" # Original: "Resources/vector"
env_file_path = "Resources/keys.env" # Original: "Resources/keys.env"

@st.cache_resource
def load_faiss_and_chat(faiss_index_path=faiss_index_path):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    load_dotenv(find_dotenv(env_file_path)) # Load .env from execution directory
    huggingfacehubapi = os.getenv('HuggingfaceRead')


    # Load FAISS vector store
    vector_store = FAISS.load_local(faiss_index_path, embedding_model,allow_dangerous_deserialization=True) # Load index from execution directory


    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",
        temperature=0.7,
        max_new_tokens=512,
        huggingfacehub_api_token=huggingfacehubapi)
    

    # Custom prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "question","chat_history"],
        template="""You are a friendly store assistant helping a customer based on the reviews and product info below. 
        If the answer is not in the context, say "I don't know." 
        Provide the requested information in a bulleted list. 
        - If asked for reviews, only return the top 3 reviews.
          - Review 1
          - Review 2 
          - Review 3
        - If asked for product names,only return the top 3 product names.
          - * Product Name 1 
          - * Product Name 2 
          - * Product Name 3 
        - If asked for price, return the price in the following format: 
          - * Product Name , Price: $xx.xx
        - If asked for rating, return the rating in the following format:
          - * Product Name Rating: x.x stars
        - If asked for color, return the color in the following format:
          - * Product Name Color: [color name]
        - If asked for category, return the category in the following format:
          - * Product Name Category: [category name]
        If there are duplicates, choose only one entry for each product. 
        Please include a more converstational tone.
        Chat History:
        {chat_history}

        Question: {question}
        Context:{context}
        Answer:""")

    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True,max_token_limit=1000)

    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                     retriever=vector_store.as_retriever(),
                                                     memory=memory,
                                                     combine_docs_chain_kwargs={"prompt" : prompt_template})
    
    return qa_chain 

qa_chain = load_faiss_and_chat()

@st.cache_data
def process_query(query):
    response = qa_chain.invoke(query)
    return (response['answer'])
