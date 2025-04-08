import pandas as pd
import json
from pprint import pprint
import os
import json
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

def load_faiss_and_chat(query,faiss_index_path=faiss_index_path):
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
        template="""You are a friendly  store assistant helping a customer based on the reviews and product info below. 
        If the answer is not in the context, say "I don't know." 
        When making recommendations, provide only the top 3 distinct products in the following format: 
        - * Product Name 1
        - * Product Name 2
        - * Product Name 3
        If there are duplicates, choose only one entry for each product.       
        Do not include any additional information like price, color, rating, or category. Only list the product names.
        Chat History:
        {chat_history}

        Question: {question}
        Context:{context}
        Answer:""")

    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)

    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                     retriever=vector_store.as_retriever(),
                                                     memory=memory,
                                                     combine_docs_chain_kwargs={"prompt" : prompt_template})
    
    response = qa_chain.invoke(query)
    return (response['answer'])
