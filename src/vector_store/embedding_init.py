import requests
import os
from pathlib import Path
from src.models.MemberMessage import MemberMessage
from src.vector_store.chroma import VectorStore
from typing import List, Optional

def fetch_member_messages() -> List[MemberMessage]:
    base_url = os.getenv("MEMBER_MESSAGES_API_URL")
    url = f"{base_url}/messages?limit=3349"
    response = requests.get(url, timeout=60.0)
    response.raise_for_status()
    data = response.json()
    messages_data = data.get("items", data.get("messages", []))
    return [MemberMessage(**msg) for msg in messages_data]

def add_member_messages_to_vector_store(messages: List[MemberMessage]) -> None:
    vector_store = VectorStore()
    vector_store.add_messages(messages)

def load_embeddings() -> Optional[VectorStore]:
    try:
        vector_store = VectorStore()
    except ValueError as e:
        return None
    
    count = vector_store.get_count()
    
    if count == 0:
        print("No data found in vector store. Fetching from API...")
        
        try:
            messages = fetch_member_messages()
            if not messages:
                print("No messages fetched from API")
                return None
            
            print("Generating embeddings and adding to vector store...")
            vector_store.add_messages(messages)
            print(f"Successfully loaded {len(messages)} messages into vector store")
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    return vector_store