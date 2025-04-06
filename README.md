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

This is a chatbot assistant using Streamlit or Gradio as a recommendation engine using Content Based Filtering to help purchase home and kitchen products on Amazon based on Amazon reviews.  Solves the problem of making recommendations without sorting through hundreds of products or going through numerous reviews before making a purchase decision, saving time and receiving product recommendations that are more fitting. 

### Project Objectives
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

#### Data Collection & Cleaning

* Sentiment Analysis - BERT Based from Hugging Face to fine tune model with above dataset
* Q/A (Hugging Face)
* Text Summarization (Hugging Face)
* Translation (Hugging Face)
* LLM (Chat GPT)

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

