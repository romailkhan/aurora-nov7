import requests
import os
from src.models.MemberMessage import MemberMessage
from src.vector_store.chroma import VectorStore
from typing import List

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