from openai import OpenAI
import os
from typing import Dict

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

