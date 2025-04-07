# Easier Buying, Happier Home: 
## Saving time and making purchasing decisions simpler in the kitchen and home

### Table of Contents

1. [Contributors](#contributors)
2. [Executive Summary](#executive-summary)
3. [Project Objectives](#project-objectives)
4. [Approach and Methodology](#approach-and-methodology)
5. [Models](#models)
6. [Tools and Techniques](#tools-and-techniques)

### Contributors

Contributors to the project:
* **Deborah Aina (Advisor)**
* **Cameron Keplinger**
* **Eshumael Manhanzva**
* **Luther Johnson**
* **Saurabh Pratap Singh**
* **Sowmya Shetty**
* **Valarie Miller**

### Executive Summary 

This project features a chatbot assistant built with Streamlit, designed to serve as a recommendation engine for home and kitchen products on Amazon. It uses content-based filtering, leveraging Amazon reviews to suggest products tailored to user preferences. The assistant addresses the challenge of browsing countless products and reviews by streamlining the decision-making process—saving time and offering more relevant recommendations. The solution includes a Question-and-Answer model integrated with a large language model (LLM), enabling a conversational interface that enhances the shopping experience through intelligent, review-based suggestions.

* **Transformer Neural Network:** The foundational architecture used for natural language processing tasks.
* **Chatbot Assistant (Built with Streamlit):** A user-friendly interface that allows interactive conversations and product recommendations.
* **DistilBERT – Question and Answering Model:** A lightweight transformer model used to extract relevant information from a subset of Amazon review data.
* **Large Language Model (LLM) – Mistral with Prompt Engineering:** Processes raw Amazon review data using tailored prompts to generate accurate and context-aware responses.


### Project Objectives
#### How does your advanced machine learning approach solve this real-world problem?:
Our Machine Learning approach addresses the challenge of finding the right product by eliminating the need to sift through hundreds of listings and reviews, saving time, and delivering more relevant, personalized recommendations.
#### The Dataset Details:
* AmazonHomeKitchenReviews.CSV
* Data source is https://amazon-reviews-2023.github.io/#grouped-by-category
* 754,079 Total Records with 18 columns
* Categories: 
  * Kids Home Store
  * Valentines Day in Home
  * Bath
  * Bedding
  * Cosmetic Organizers
  * Dorm Room HQ
  * Event & Party Supplies
  * Furniture
  * Heating, Cooling & Air Quality
  * Home and Furniture Made in Italy
  * Home Decor Products
  * Irons & Steamers
  * Kitchen & Dining
  * Seasonal Decor
  * Storage & Organization
  * Vacuums & Floor Care
  * Wall Art
  * Small Appliance Parts

#### Data Modeling Strategy
This system combines advanced Natural Language Processing techniques (NLP) to deliver intelligent, conversational product recommendations:

* **Sentiment Analysis (BERT-based, Hugging Face):** Uses a fine-tuned BERT model to analyze the sentiment of Amazon product reviews, enhancing the quality of recommendations.
* **Question and Answering (DistilBERT via Hugging Face):** Implements DistilBERT to extract precise answers from review data, enabling the chatbot to respond to user queries effectively.
* **Text Summarization (via LLM):** Summarizes large volumes of review content to present concise, relevant insights to the user.
* **LangChain:** Used to manage the flow of conversation and integrate various LLM components seamlessly.
* **PyTorch Library:** Core machine learning framework used for training and deploying custom models.
* **RAG (Retrieval-Augmented Generation):** Enhances the chatbot's responses by integrating a custom knowledge base and vector data stores for more accurate, context-aware answers.
* **Translation (TBD):** Potential future feature to support multilingual users by translating queries and responses.

### Approach and Methodology

* Our approach utilizes the following: 
  * Load the CSV data file
  * Load the api key for HuggingfaceRead
  * Load all the records in the dataframe as documents using the load_docs function
  * Create the path for the vector database. 
  * Call the store_incrementally_in_fiass function
  * Create the function to load the vector database and chat (using HuggingFaceEndpoint mistralai/Mistral-7B-Instruct-v0.1 as the LLM) 
  * Create a question answer retreival chain from langchain.chains framework

#### Exploratory Data Analysis

The dataset covers a wide age range of....

###  Models

The models used were....

### Tools and Techniques

* Pandas & Pandas Plotting
* Python
* Scikit-learn
* Matplotlib
* Numpy

