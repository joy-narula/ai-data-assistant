from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI
import streamlit as st
import os 
from dotenv import load_dotenv

pinecone_key = st.secrets["PINECONE_API_KEY"]
# Loading env variables
load_dotenv()
openai_key = st.secrets["OPENAI_API_KEY"]

pc = Pinecone(pinecone_key)
client = OpenAI(api_key=openai_key)

index = pc.Index('ai-assistant')
model = SentenceTransformer('all-MiniLM-L6-v2')

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(vector=input_em, top_k=2, include_metadata=True)
    return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']

def query_refiner(conversation, query):
    response = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that refines user queries for better knowledge base answers."},
            {"role": "user", "content": f"CONVERSATION LOG:\n{conversation}"},
            {"role": "user", "content": f"Query: {query}\n\nRefined Query:"},
        ],
        temperature = 0.7,
        max_tokens = 256,
        top_p = 1,
        frequency_penalty = 0,
        presence_penalty = 0
    )
    
    return response.choices[0].message.content.strip()

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i+1] + "\n"
    return conversation_string
