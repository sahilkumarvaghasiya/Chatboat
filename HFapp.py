from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import HuggingFaceChat
from langserve import add_routes
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Initialize FastAPI
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server"
)

# Load the local model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Wrap the local model with LangChain
class LocalHuggingFaceChat(HuggingFaceChat):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt, max_new_tokens=100, **kwargs):
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        outputs = self.model.generate(inputs, max_new_tokens=max_new_tokens, **kwargs)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

local_model = LocalHuggingFaceChat(model, tokenizer)

# Define prompts
prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2 = ChatPromptTemplate.from_template("Write me a poem about {topic} for a 5-year-old child with 100 words")

# Add routes
add_routes(
    app,
    prompt1 | local_model,
    path="/essay"
)

add_routes(
    app,
    prompt2 | local_model,
    path="/poem"
)

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)


    
