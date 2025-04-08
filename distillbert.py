from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from transformers import pipeline
import torch 


your_model_dir = "Resources/distillbert"

# Load model and tokenizer from a local folder containing safetensors

def load_model():
    model = AutoModelForQuestionAnswering.from_pretrained(your_model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(your_model_dir)
    return pipeline("question-answering", model=model, tokenizer=tokenizer)


qa_pipeline = load_model()

# --- Function to Get Answer ---
def get_distilbert_answer(query: str,context: str) -> str:
    """
    Use the fine-tuned DistilBERT model to answer the question based on the context.
    """
    result = qa_pipeline(question=query, context=context)

    if result and result["score"] > 0.1:
        return result["answer"]
    else:
        return "Sorry, I couldn't find a confident answer based on the context."