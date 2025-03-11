from flask import Flask, render_template, request, jsonify
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pickle
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("faiss_index.bin")

with open("text_chunks.pkl", "rb") as f:
    split_texts = pickle.load(f)

text_splitter = RecursiveCharacterTextSplitter()

client = OpenAI(api_key = api_key)

def retrieve(query, top_k=3):
    query_embedding = model.encode(query, convert_to_tensor=False).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    results = [split_texts[i] for i in indices[0]]
    return results

def generate_answer(query):
    retrieved_texts = retrieve(query)
    context = "\n\n".join(retrieved_texts)
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
        {"role": "user", "content": f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}"}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=200
    )
    
    return response.choices[0].message.content.strip()

@app.route('/')
def homePage():
    return render_template('index.html') 

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    # response = generate_answer(user_message) # Uncomment this line in case you are using any LLM model setting the API Key in .env file. 
    response = retrieve(user_message) #Comment this line if you are using above line for a response. 
    return jsonify({"message":response})

if __name__ == "__main__":
    app.run(debug=True, port=5398)