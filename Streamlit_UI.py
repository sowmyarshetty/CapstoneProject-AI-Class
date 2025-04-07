 --- Streamlit UI Elements ---

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
st.markdown('<div class="header-container"><h1>Amazon Review Assistant <span>QA</span></h1></div>', unsafe_allow_html=True)

# Add sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # GPT-4o Integration
    st.subheader("GPT-4o Integration")
    # Store configuration in session state instead of global variables
    if 'use_gpt4o' not in st.session_state:
        st.session_state.use_gpt4o = USE_GPT4O
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = OPENAI_API_KEY
        
    # UI controls
    st.session_state.use_gpt4o = st.toggle("Use GPT-4o for enhanced responses", value=st.session_state.use_gpt4o)
    
    # API Key input with proper security
    st.session_state.openai_api_key = st.text_input("OpenAI API Key (required for GPT-4o)", 
                           type="password", 
                           value=st.session_state.openai_api_key,
                           help="Your API key will not be stored permanently")
    
    if st.session_state.use_gpt4o and not st.session_state.openai_api_key:
        st.warning("⚠️ Please enter your OpenAI API key to use GPT-4o")
    
    if st.session_state.use_gpt4o and st.session_state.openai_api_key:
        st.success("✅ GPT-4o integration is enabled")
    
    st.divider()
    
    st.markdown("""
    ### About
    This app uses a trained QA model to extract answers from Amazon product reviews.
    
    When GPT-4o integration is enabled, the app will use OpenAI's GPT-4o model to enhance 
    the extracted answers, making them more natural and comprehensive.
    
    Your API key is used only for API calls and is not stored permanently.
    """)

st.markdown("""
Ask questions about home and kitchen products. I'll find relevant reviews and use a **trained QA model** to extract answers directly from the text.
*Example: "How noisy is the XYZ blender according to reviews?"*
""")

# --- Chat Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you find answers within product reviews?"}]

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if user_query := st.chat_input("Ask a question based on reviews..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Process and get bot response
    with st.chat_message("assistant"):
        response_placeholder = st.empty() # Use a placeholder for streaming-like effect
        with st.spinner("Searching reviews and extracting answer..."):
            try:
                # 1. Retrieve relevant documents
                retrieved_docs = retriever.invoke(user_query)
                context = "\n\n".join([doc.page_content for doc in retrieved_docs]) # Combine content

                if not context:
                    response = "I couldn't find relevant reviews for that question."
                    logger.warning(f"No context found for query: {user_query}")
                else:
                    # 2. Get answer from context using the local QA model
                    response = get_answer_from_context(user_query, context, tokenizer, qa_model)

                response_placeholder.markdown(response) # Display the final response
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                logger.exception(f"Error processing user query '{user_query}': {e}")
                error_message = f"Sorry, an error occurred: {e}"
                response_placeholder.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
