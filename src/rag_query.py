import numpy as np
from typing import List
from supabase import Client
import ollama
from dotenv import load_dotenv
import os

load_dotenv()

def get_relevant_chunks(
    query: str, 
    client: Client,
    chat_model: str,
    top_k: int = 10
) -> str:
    """Retrieve top-K most relevant transaction chunks for a query"""
    
    # Determine host and model name
    if "::" in chat_model:
        instance, model_name = chat_model.split("::")
        if instance == "chat":
            host = os.getenv("OLLAMA_HOST_CHAT")
        else: # Should not happen based on UI filtering
            host = os.getenv("OLLAMA_HOST_OCR")
    else: # No prefix, use default host
        model_name = chat_model
        host = os.getenv("OLLAMA_HOST")

    ollama_client = ollama.Client(host=host)

    # Generate query embedding
    try:
        query_embedding = ollama_client.embeddings(
            model='nomic-embed-text', 
            prompt=query
        )['embedding']
        query_embedding = np.array(query_embedding)
    except Exception as e:
        print(f"Embedding generation failed: {e}")
        return ""
    
    # Fetch all transaction embeddings
    try:
        response = client.table('transactions').select('id, full_chunk, embedding').execute()
        if not response.data:
            return ""
        
        chunks = [item['full_chunk'] for item in response.data]
        embeddings = np.array([item['embedding'] for item in response.data])
        
        # Calculate cosine similarities
        similarities = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top K indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return concatenated context
        context = "\n".join([chunks[i] for i in top_indices])
        return context
        
    except Exception as e:
        print(f"RAG retrieval failed: {e}")
        return ""

def generate_insights(query: str, context: str, model: str) -> str:
    """Generate insights using retrieved context"""
    
    # Determine host and model name
    if "::" in model:
        instance, model_name = model.split("::")
        if instance == "chat":
            host = os.getenv("OLLAMA_HOST_CHAT")
        else: # Should not happen based on UI filtering
            host = os.getenv("OLLAMA_HOST_OCR")
    else: # No prefix, use default host
        model_name = model
        host = os.getenv("OLLAMA_HOST")

    ollama_client = ollama.Client(host=host)

    prompt = f"""You are a personal finance assistant. Based on the following transaction data, 
    provide helpful insights for the user's query. Cover categorization, spending trends, 
    anomalies, budgeting advice, and potential tax deductions where relevant.

    Transaction Data:
    {context}

    User Query: {query}

    Provide a concise, actionable response focusing on:
    1. Key patterns or trends
    2. Spending categorization
    3. Any anomalies or unusual activity
    4. Budgeting recommendations
    5. Potential tax-deductible expenses

    Response:"""
    
    try:
        response = ollama_client.generate(model=model_name, prompt=prompt)
        return response['response'].strip()
    except Exception as e:
        return f"Error generating insights: {e}"