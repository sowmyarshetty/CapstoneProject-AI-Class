import streamlit as st
from llmmodel import process_query # For Mistral tab
from distillbert import get_distilbert_answer # For DistilBERT tab
import torch
import os
from PIL import Image
import base64
# --- Helper function to load and encode the image ---
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()
# Get the path to ZonBot image
image_path = os.path.join("Resources", "ZonBot.jpg")
if os.path.exists(image_path):
    encoded_image = get_base64_image(image_path)
else:
    st.error(f"Image not found at {image_path}")
    encoded_image = ""

# --- Streamlit UI Elements ---

# Inject Custom CSS for Amazon Look
st.markdown(
    """
<style>
    /* Overall app background and text color */
    .stApp {
        background-color: #0A0F1C; /* Deep space blue */
        color: #FFFFFF; /* Global white text */
    }
    /* Header container setup with background image */
    .header-container {
        background-color: transparent;
        padding: 0;
        margin-bottom: 40px; /* Space below header */
        color: #FFFFFF;
        display: flex;
        flex-direction: column;
        align-items: center;
        position: relative;
    }
    /* ZonBot banner image */
    .header-container::before {
        content: "";
        width: 100%;
        max-width: 720px;
        height: 236px;
        background-image: url('data:image/jpeg;base64,""" + encoded_image + """');
        background-size: cover;
        background-position: center;
        border-radius: 8px;
        box-shadow: 0 0 12px #0F50EC;
        margin-bottom: 20px;
    }
    /* Header title */
    .header-container h1 {
        color: #FFFFFF;
        font-size: 1.8em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    /* AI badge in the header */
    .header-container h1 span {
        color: #39FF14;
        font-weight: bold;
        font-size: 0.8em;
        margin-left: 8px;
        background-color: #3A0CA3;
        padding: 2px 6px;
        border-radius: 4px;
    }
    /* Chat messages container */
    [data-testid="stChatMessage"] {
        border-radius: 15px;
        padding: 0.75rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        border: 1px solid #7D3C98;
    }
    /* Assistant message bubble */
    [data-testid="stChatMessage"][data-testid="chatAvatarIcon-assistant"] > div > div {
        background-color: #1B263B;
        color: #FFFFFF;
    }
    /* User message bubble */
    [data-testid="stChatMessage"][data-testid="chatAvatarIcon-user"] > div > div {
        background-color: #3D2C8D;
        color: #FFFFFF;
    }
    /* Input area for user interaction */
    [data-testid="stChatInput"] {
        background-color: #0D1B2A;
        border-top: 1px solid #34495E;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.3);
        color: #FFFFFF;
    }
    /* Input area for user interaction */
    [data-testid="stChatInputTextArea"] {
        background-color: #0D1B2A;
        border-top: 1px solid #34495E;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.3);
        color: #FFFFFF;
    }
    /* Button styling */
    .stButton>button {
        background-color: #39FF14;
        color: #0A0F1C;
        border: 1px solid #39FF14;
        border-radius: 8px;
    }
    .text1 { color: "#FFFFFF"}
    /* Make sure other text elements are also white */
    h2, h3, h4, h5 ,p,span,div{
        color: #FFFFFF;
    }
     
    /* Target the main select box container */
    div[data-baseweb="select"] {
        background-color: #000000;
        border-radius: 8px;
        padding: 5px;
        color: white;
    }

    /* Style the selected value */
    div[data-baseweb="select"] div[role="button"] {
        color: white;
        background-color: #000000;
        font-weight: bold;
    }

    /* Optional: Style the dropdown list (on click) */
    ul[role="listbox"] {
        background-color: #000000;
        color: white;
    }

    li[role="option"] {
        color: #000000;
        }
        
    /* Optional: Change hover effect on dropdown options */
    li[role="option"]:hover {
        background-color: #000000;
        color: white;
    }
   
</style>
""",
    unsafe_allow_html=True,
)

# Custom Header Display
st.markdown('<div class="header-container"><h1>Review Assistant <span>AI</span></h1></div>', unsafe_allow_html=True)
st.markdown('<div class="text1"><h4> Ask questions about home and kitchen products based on reviews.</h4></div>',unsafe_allow_html=True)

# --- Initialize Session State for Chat Histories ---
if "messages_tuned" not in st.session_state:
    st.session_state.messages_tuned = [{"role": "assistant", "content": "Hi! Ask me anything about the product reviews Distillbert -Fine-Tuned Model"}]
if "messages_base" not in st.session_state:
    st.session_state.messages_base = [{"role": "assistant", "content": "Hi! Ask me anything about the product- LLM (Mistral))."}]

# --- Create Tabs ---
# Renaming tabs as requested
tab1, tab2 = st.tabs(["Fine-Tuned Model (DistilBERT)", "LLM (Mistral)"])

# --- Tab 1: Base Model (DistilBERT) ---
with tab1:
    st.header("Chat with finetuned Model (DistilBERT)")

    # Display chat messages (using messages_tuned for this tab now)
    for message in st.session_state.messages_tuned:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if user_query_tuned := st.chat_input('Ask the finetuned DistilBERT model...'):
        # Add user message to history and display
        st.session_state.messages_tuned.append({"role": "user", "content": user_query_tuned})
        with st.chat_message("user"):
            st.markdown(user_query_tuned)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            with st.spinner("Thinking..."):
                try:
                    # Call the function from distillbert.py
                    final_answer = get_distilbert_answer(user_query_tuned)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    final_answer = "Sorry, I encountered an issue processing your request."

                # Display final answer
                response_placeholder.markdown(final_answer)
                # Append to the correct history
                st.session_state.messages_tuned.append({"role": "assistant", "content": final_answer})

# --- Tab 2: Fine-Tuned Model (Mistral) ---
with tab2:
    st.header("Chat with LLM (Mistral)")

    # Display chat messages (using messages_base for this tab now)
    for message in st.session_state.messages_base:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if user_query_base := st.chat_input("Ask the Mistral model..."):
        # Add user message to history and display
        st.session_state.messages_base.append({"role": "user", "content": user_query_base})
        with st.chat_message("user"):
            st.markdown(user_query_base)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            with st.spinner("Thinking..."):
                try:
                    # Call the function from llmmodel.py (Mistral)
                    final_answer = process_query(user_query_base)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    final_answer = "Sorry, I encountered an issue processing your request."

                # Display final answer
                response_placeholder.markdown(final_answer)
                # Append to the correct history
                st.session_state.messages_base.append({"role": "assistant", "content": final_answer})


# --- Sidebar ---
with st.sidebar:
    st.header("Chat Controls")
    if st.button("Clear All Chat History"):
        st.session_state.messages_tuned = [{"role": "assistant", "content": "Chat history cleared. Ask me a new question!"}]
        st.session_state.messages_base = [{"role": "assistant", "content": "Chat history cleared. Ask me a new question!"}]
        st.rerun()
