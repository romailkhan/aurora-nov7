import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any
from openai import OpenAI
from src.models.MemberMessage import MemberMessage
import os

class VectorStore:
    """Manages ChromaDB vector store for member messages."""
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = OpenAI(api_key=api_key)
        
        self.chroma_client = chromadb.PersistentClient(
            path="../../data/chroma_db",
            settings=ChromaSettings(
                anonymized_telemetry=False
            )
        )
        
        self.collection = self.chroma_client.get_or_create_collection(
            name="member_messages",
            metadata={"description": "Aurora member messages"}
        )
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using OpenAI embedding model.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.openai_client.embeddings.create(
                model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
                input=batch
            )
            all_embeddings.extend([item.embedding for item in response.data])
        
        return all_embeddings
    
    def add_messages(self, messages: List[MemberMessage]) -> None:
        """
        Add member messages to the vector store.
        
        Args:
            messages: List of MemberMessage objects to add
        """
        if not messages:
            return
        
        documents = [msg.to_document_text() for msg in messages]
        metadatas = [msg.to_metadata() for msg in messages]
        ids = [msg.id for msg in messages]
        
        embeddings = self._generate_embeddings(documents)
        
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
    
    def search(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """
        Search for relevant messages using semantic search.
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            
        Returns:
            Dictionary with results, metadatas, and distances
        """
        if top_k is None:
            top_k_env = os.getenv("TOP_K")
            top_k = int(top_k_env) if top_k_env else 5
        
        query_embedding = self._generate_embeddings([query])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        return results
    
    def clear(self) -> None:
        """Clear all data from the collection."""
        self.chroma_client.delete_collection("member_messages")
        self.collection = self.chroma_client.get_or_create_collection(
            name="member_messages",
            metadata={"description": "Aurora member messages"}
        )
    
    def get_count(self) -> int:
        """Get the number of messages in the vector store."""
        return self.collection.count()
    
    def refresh_data(self, messages: List[MemberMessage]) -> None:
        """
        Refresh the vector store with new data.
        
        Args:
            messages: New list of messages to replace existing data
        """
        self.clear()
        self.add_messages(messages)

