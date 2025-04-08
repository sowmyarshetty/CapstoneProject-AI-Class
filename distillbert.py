import streamlit as st
from transformers import DistilBertForQuestionAnswering
from transformers import DistilBertTokenizerFast
from transformers import pipeline
# question_answerer = pipeline("question-answering", model='distilbert-base-uncased-distilled-squad')

your_model_dir = "Resources\distillbert"

# Load model and tokenizer from a local folder containing safetensors
@st.cache_resource
def load_model():
    model = DistilBertForQuestionAnswering.from_pretrained("your_model_dir", trust_remote_code=True)
    tokenizer = DistilBertTokenizerFast.from_pretrained("your_model_dir")
    return pipeline("question-answering", model=model, tokenizer=tokenizer)


# #This is for the  UI
# st.title("Text Classification with Safetensors Model")
# user_input = st.text_input("Enter your text:")

# if user_input:
#     classifier = load_model()
#     prediction = classifier(user_input)
#     st.write("Prediction:", prediction)
