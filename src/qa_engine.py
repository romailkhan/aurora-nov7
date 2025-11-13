from openai import OpenAI
import os
from typing import Dict, Tuple, Optional
from src.vector_store.chroma import VectorStore

def query(query: str, vector_store: Optional[VectorStore] = None, top_k: int = 5) -> Tuple[str, Dict]:
    """
    Query the Q&A system with a question.
    
    Args:
        query: The question to ask
        vector_store: Optional VectorStore instance. If None, will be initialized.
        top_k: Number of search results to use (default: 5)
        
    Returns:
        Tuple of (answer: str, results: Dict) where results contains search metadata
    """
    if vector_store is None:
        from src.vector_store.embedding_init import load_embeddings
        vector_store = load_embeddings()
        if vector_store is None:
            return "Failed to initialize vector store.", {}
    
    search_results = vector_store.search(query, top_k=top_k)
    answer = generate_answer(query, search_results)
    
    return answer, search_results

def generate_answer(query: str, search_results: Dict) -> str:
    if not search_results.get("metadatas") or not search_results["metadatas"][0]:
        return "I don't have any information to answer this question."
    
    metadatas = search_results["metadatas"][0]
    
    context_parts = []
    for i, metadata in enumerate(metadatas):
        context_parts.append(
            f"[Message {i+1}]\n"
            f"User: {metadata['user_name']}\n"
            f"Date: {metadata['timestamp']}\n"
            f"Message: {metadata['message']}\n"
        )
    
    context = "\n".join(context_parts)
    
    system_prompt = """You are a helpful assistant that answers questions about member data.

    You will be provided with relevant member messages and a question. Your task is to:
    1. Analyze the provided messages carefully
    2. Answer the question based on the information in the messages
    3. Be specific and cite user names when relevant
    4. CRITICAL: Currently, there is no person named Amira, therefore if the question is about Amira, you should say that you don't have any information about her.
    5. If the information is not in the messages, provide a helpful response explaining what you can or cannot determine
    6. Keep answers concise and direct.
    7. Do not mention sources, citations, or which messages the information came from.
    8. If you are given a question about a person not in the data, acknowledge this naturally without being overly formal."""

    user_prompt = f"""Context (relevant member messages):
    {context}

    Question: {query}

    Please provide a clear, concise answer based on the information above."""

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=500
    )
    
    return response.choices[0].message.content.strip()

def display_answer(answer: str, query: str, results: Dict):
    print(f"\nQuestion: {query}")
    print("=" * 80)
    print(f"\nAnswer:\n{answer}\n")
    print("=" * 80)
    
    if not results.get("metadatas") or not results["metadatas"][0]:
        return
    
    metadatas = results["metadatas"][0]
    distances = results["distances"][0] if results.get("distances") else []
    
    print(f"\nSources ({len(metadatas)} messages):\n")
    
    for i, metadata in enumerate(metadatas):
        distance = distances[i] if distances else None
        relevance = 1.0 / (1.0 + distance) if distance is not None else None
        
        print(f"[Source {i+1}]")
        print(f"User: {metadata.get('user_name', 'Unknown')}")
        print(f"Date: {metadata.get('timestamp', 'Unknown')}")
        print(f"Message: {metadata.get('message', 'N/A')}")
        if relevance is not None:
            print(f"Relevance: {relevance:.3f}")
        print("-" * 80)

