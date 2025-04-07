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
st.markdown("Ask questions about home and kitchen products. I'll find relevant reviews, extract key info using a **local QA model**, and then use **Mistral** to give you a conversational answer.")

# --- Initialize Session State for Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you find answers within product reviews?"}]

# --- Display Chat Messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle User Input and Generate Response ---
if user_query := st.chat_input("Ask a question based on reviews..."):
    # Add user message to history and display
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        with st.spinner("Searching reviews, extracting info, and generating answer..."):
            final_answer = "Sorry, I encountered an issue processing your request." # Default error
            try:
                # 1. Retrieve Context
                if not retriever:
                    st.error("Review retriever is not available. Cannot process query.")
                    # Use st.stop() cautiously, maybe just set final_answer and log
                    final_answer = "Error: Review retriever not initialized."
                    logger.error("Retriever not available.")
                else:
                    retrieved_docs = retriever.invoke(user_query)
                    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {user_query}")

                    # 2. Extract Facts (Local QA Model)
                    extracted_info = "Could not extract specific information from reviews." # Default
                    if context:
                        if local_tokenizer and local_qa_model:
                            extracted_info = get_answer_from_context(
                                user_query,
                                context,
                                local_tokenizer,
                                local_qa_model
                            )
                        else:
                            extracted_info = "Local QA model is not available for extraction."
                            logger.warning("Local QA model/tokenizer not loaded, skipping extraction.")
                    else:
                        extracted_info = "No relevant reviews found to extract information from."
                        logger.warning(f"No context retrieved for query: {user_query}")

                    # 3. Synthesize Answer (Mistral LLM)
                    # Only proceed if retriever worked
                    synthesis_prompt = f"""Based on the following information extracted from product reviews:
"{extracted_info}"

Please answer the user's original question in a helpful and conversational way: "{user_query}"

If the extracted information indicates it couldn't find an answer, no reviews were found, or the local model wasn't available, please state that clearly. Avoid making up information not present in the extracted text.
Answer:
"""
                    logger.info(f"Sending synthesis prompt to Mistral:\n{synthesis_prompt}")

                    mistral_response = mistral_llm.invoke(synthesis_prompt)
                    final_answer = mistral_response.strip() if isinstance(mistral_response, str) else str(mistral_response)

            except Exception as e:
                logger.exception(f"Error processing user query '{user_query}': {e}")
                final_answer = f"Sorry, an error occurred: {e}" # Show error in chat

            # Display final answer
            response_placeholder.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

# --- Sidebar ---
with st.sidebar:
    st.header("Chat Controls")
    if st.button("Clear Chat History"):
        st.session_state.messages = [{"role": "assistant", "content": "Chat history cleared. Ask me a new question!"}]
        st.rerun()

