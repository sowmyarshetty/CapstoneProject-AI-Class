import streamlit as st
from llmmodel import process_query # For Mistral tab
from distillbert import get_distilbert_answer # For DistilBERT tab
import torch 

# --- Streamlit UI Elements ---

# Inject Custom CSS for Amazon Look
st.markdown(
    """
<style>
    /* Target the main block container */
    .stApp {
        /* background-color: #EAEDED; */ /* Uncomment for light grey background */
    }

    /* Header */
    .header-container {
        background-color: #232F3E;
        padding: 10px 20px;
        border-radius: 5px;
        margin-bottom: 20px;
        color: white;
    }
    .header-container h1 {
        color: white;
        margin: 0;
        font-size: 1.8em;
        display: flex;
        align-items: center;
    }
    .header-container h1 span {
        color: #FF9900;
        font-weight: bold;
        font-size: 0.8em; /* Smaller 'AI' */
        margin-left: 8px;
        background-color: #4a5a6a; /* Darker badge background */
        padding: 2px 6px;
        border-radius: 4px;
    }
    /* Custom Amazon-like cart icon (using emoji) */
     .header-container h1::before {
        content: "🛒"; /* Cart Emoji */
        margin-right: 10px;
        font-size: 1.2em;
     }


    /* Chat Messages */
    [data-testid="stChatMessage"] {
        border-radius: 15px;
        padding: 0.75rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #D5DBDB; /* Subtle border */
    }
    [data-testid="stChatMessage"][data-testid="chatAvatarIcon-assistant"] > div > div {
         background-color: #F1F1F1; /* Light grey for bot messages */
         color: #0F1111;
    }
    [data-testid="stChatMessage"][data-testid="chatAvatarIcon-user"] > div > div {
         background-color: #E3F2FD; /* Light blue for user messages */
         color: #0F1111;
    }

    /* Input Area */
    [data-testid="stChatInput"] {
        background-color: #FFFFFF;
        border-top: 1px solid #CCCCCC;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.05);
    }
     /* Button styling (partially from theme, can override) */
    .stButton>button {
        /* background-color: #FF9900; */ /* Theme primaryColor should handle this */
        /* color: #FFFFFF; */
        /* border: 1px solid #FF9900; */
        border-radius: 8px; /* Slightly more rounded */
    }

</style>
""",
    unsafe_allow_html=True,
)

# Custom Header Display
st.markdown('<div class="header-container"><h1>Review Assistant <span>AI</span></h1></div>', unsafe_allow_html=True)
st.markdown("Ask questions about home and kitchen products based on reviews.")

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
    if user_query_base := st.chat_input("Ask the finetuned DistilBERT model..."):
        # Add user message to history and display
        st.session_state.messages_tuned.append({"role": "user", "content": user_query_base})
        with st.chat_message("user"):
            st.markdown(user_query_base)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            with st.spinner("Thinking..."):
                try:
                    # Call the function from distillbert.py
                    final_answer = get_distilbert_answer(user_query_base,user_query_base)
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

    # Display chat messages (using messages_tuned for this tab now)
    for message in st.session_state.messages_base:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if user_query_tuned := st.chat_input("Ask the Mistral model..."):
        # Add user message to history and display
        st.session_state.messages_base.append({"role": "user", "content": user_query_tuned})
        with st.chat_message("user"):
            st.markdown(user_query_tuned)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            with st.spinner("Thinking..."):
                try:
                    # Call the function from llmmodel.py (Mistral)
                    final_answer = process_query(user_query_tuned)
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
