
import google.generativeai as palm
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# model_name = "gemini-pro"

# Set LangChain API key
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
api_key = os.getenv("PALM_API_KEY")
palm.configure(api_key=api_key)

# Define ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("user", "Question:{question}")
])

# Streamlit setup
st.title('Langchain Demo With Google Palm Model')
input_text = st.text_input("Search the topic you want")


#create function to get response
def generate_response(question):
    response = palm.generate_text(
        model='models/text-bison-001',
        prompt=formatted_input,
        max_output_tokens=100  # Adjust as needed
    )
    return response.result

output_parser = StrOutputParser()

if input_text:
    formatted_input = prompt.format(question=input_text)
    response = generate_response(formatted_input)
    parsed_response = output_parser.parse(response)
    st.write(parsed_response)