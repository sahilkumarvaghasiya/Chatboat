from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from langserve import add_routes
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Langchain-Server",
    version="1.0",
    description="A simple API server"
)

# Set up Hugging Face API token
huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN")

# Initialize Hugging Face Endpoint model
hf_endpoint = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/gpt2",  # Replace with your desired model endpoint
    api_token=huggingface_api_token
)

# Initialize ChatHuggingFace model
model = ChatHuggingFace(llm=hf_endpoint)

# Define prompts
prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2 = ChatPromptTemplate.from_template("Write me a poem about {topic} for a 5 years child with 100 words")

# Add routes to FastAPI app
add_routes(
    app,
    prompt1 | model,
    path="/essay"
)

add_routes(
    app,
    prompt2 | model,
    path="/poem"
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
