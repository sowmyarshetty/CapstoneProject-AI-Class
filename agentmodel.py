import os
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from typing import List, Optional
from langchain.tools import Tool
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import LLMMathChain
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import initialize_agent
import streamlit as st
from dotenv import load_dotenv, find_dotenv

#replace this with your own api key from .env file
env_file_path = "Resources/keys.env" # Original: 

load_dotenv(find_dotenv(env_file_path)) # Load .env from execution directory
huggingfacehubapi = os.getenv('HuggingfaceRead')

faiss_path = os.path.join("Resources/vector")

# Initialize the Mistral model with updated parameters
llm_model = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", 
    temperature=0.1,
    max_new_tokens=512,
    huggingfacehub_api_token=huggingfacehubapi,
    model_kwargs={
        "stop": ["Human:", "Assistant:"],  
    }
    
)

# Create your tools
math_chain = LLMMathChain.from_llm(llm=llm_model)
math_tool = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Useful for performing mathematical calculations."
)

def health_redirect(query: str) -> str:
    """Redirects users to home and kitchen topics only."""
    return "I am an AI assistant that can only answer questions about home and kitchen products. Please ask about those topics."

health_tool = Tool(
    name="HealthRedirect",
    func=health_redirect,
    description="Use this when users ask about topics outside home and kitchen products."
)

# Assuming your FAISS vector store is initialized
faiss_vectorstore = FAISS.load_local(faiss_path, llm_model, allow_dangerous_deserialization=True)
retriever = faiss_vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

#Load our responses
faiss_tool = create_retriever_tool(
    retriever,
    name="ProductKnowledge",
    description="Use this to answer questions about home and kitchen products. Input should be a specific product-related question."
)

tools = [math_tool, health_tool, faiss_tool]


# Create a memory instance
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="output"
)

# Custom output parser with sign-off
class ZonbotResponse(BaseModel):
    answer: str
    product_id: str
    product_name: str
    rating: float
    products: Optional[List[str]] = Field(None, description="List of relevant products mentioned")
    
output_parser = PydanticOutputParser(pydantic_object=ZonbotResponse)

# Create a prompt template with ALL required variables
prompt_template = """You are ZonBot, a helpful AI assistant specialized in home and kitchen products.
Please respond like a real customer service in store would respond

You have access to the following tools: {tool_names}

{tools}

Use these tools to answer the user's question.

Chat History:
{chat_history}

Remember:
- Only discuss home and kitchen products
- Be specific and accurate
- If you don't know the answer, admit it
- Consider previous interactions in your responses
- Return the answer in bullet points
- ALWAYS end your response by asking if there's anything else you can help with and sign off with "ZonBot"

Use the following format:

Thought: The user is asking about {input}. 
Action: Call {tool_name} with the input {input}
Action Input: {input}
Observation: Here’s the result of the action.
Thought: Based on this observation, I now know the final answer.
Final Answer: {final_answer}

"""

prompt = PromptTemplate.from_template(prompt_template)


agent_executor = initialize_agent(
    tools=tools,
    llm=llm_model,
    agent="chat-conversational-react-description",
    handle_parsing_errors=True,
    max_iterations=3,
    verbose=True,
    memory=memory,
    prompt=prompt ) # Adding the prompt parameter here)



# Custom output formatter function to add the sign-off
def format_output(result):
    base_output = result["output"]
    
    # Check if the output already has the sign-off
    if "Is there anything else I can help you with?" not in base_output and "ZonBot" not in base_output.split("\n")[-1]:
        formatted_output = f"{base_output}\n\n Is there anything else I can help you with? \n\nZonBot"
    else:
        formatted_output = base_output
    
    try:
        # Try to parse as structured response
        response_dict = {
            "answer": formatted_output,
            "products": None,
            "recommendation": None,
            "source": None
        }
        return ZonbotResponse(**response_dict)
    except Exception:
        # Fallback to simple string output
        return {"Here is what I found": formatted_output}

# Add a check for greetings or empty inputs
def is_valid_query(query: str) -> bool:
    """Checks if the query is a valid question and not a greeting or empty input."""
    greetings = ["hello", "hi", "hey", "howdy", "greetings"]
    if not query or any(greeting in query.lower() for greeting in greetings):
        return False
    return True


# In your application
def process_user_query(query: str):
    try:
        result = agent_executor.invoke({"input": query})
        return format_output(result)
    except Exception as e:
        # Fallback for any errors during processing
        return {"answer": f"I encountered an issue while processing your request. Please try asking in a different way.\n\nIs there anything else I can help you with?\n\nZonBot"}


#process_user_query("Can you find the best ice cream maker?")