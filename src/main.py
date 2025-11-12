from pathlib import Path
from dotenv import load_dotenv
from src.vector_store.embedding_init import fetch_member_messages
from src.vector_store.chroma import VectorStore

env_path1 = Path(__file__).parent.parent / ".env"
load_dotenv(env_path1)

def clear_vector_store():
    """Clear all data from the vector store."""
    print("🗑️  Clearing vector store...")
    try:
        vector_store = VectorStore()
        vector_store.clear()
        print("Vector store cleared successfully")
        return True
    except Exception as e:
        print(f"Error clearing vector store: {e}")
        return False

def load_embeddings(force_refresh: bool = False):
    """Load embeddings from API and add to vector store."""
    print("Starting Aurora Q&A System...")
    
    try:
        vector_store = VectorStore()
    except ValueError as e:
        return None
    
    count = vector_store.get_count()
    
    if count == 0 or force_refresh:
        if force_refresh and count > 0:
            print(f"Refreshing data (current: {count} messages)...")
            vector_store.clear()
        else:
            print("No data found in vector store. Fetching from API...")
        
        try:
            print("Fetching all messages from API...")
            messages = fetch_member_messages()
            if not messages:
                print("No messages fetched from API")
                print("   Check that MEMBER_MESSAGES_API_URL is set correctly in your .env file")
                return None
            
            print(f"Fetched {len(messages)} messages from API")
            print("Generating embeddings and adding to vector store...")
            print("   (This may take several minutes depending on the number of messages)...")
            vector_store.add_messages(messages)
            print(f"Successfully loaded {len(messages)} messages into vector store")
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        print(f"Vector store already contains {count} messages")
    
    return vector_store


def display_search_results(results: dict, query: str):
    """Display search results in a readable format."""
    if not results.get("metadatas") or not results["metadatas"][0]:
        print("\nNo results found")
        return
    
    metadatas = results["metadatas"][0]
    distances = results["distances"][0] if results.get("distances") else []
    
    print(f"\nFound {len(metadatas)} relevant messages for: '{query}'\n")
    print("=" * 80)
    
    for i, metadata in enumerate(metadatas):
        distance = distances[i] if distances else None
        relevance = 1.0 / (1.0 + distance) if distance is not None else None
        
        print(f"\n[Result {i+1}]")
        print(f"User: {metadata.get('user_name', 'Unknown')}")
        print(f"Date: {metadata.get('timestamp', 'Unknown')}")
        print(f"Message: {metadata.get('message', 'N/A')}")
        if relevance is not None:
            print(f"Relevance Score: {relevance:.3f} (distance: {distance:.4f})")
        print("-" * 80)


def main():
    """Main entry point with terminal query interface."""
    import sys
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--clear":
            clear_vector_store()
            return
        elif sys.argv[1] == "--refresh":
            vector_store = load_embeddings(force_refresh=True)
            if vector_store is None:
                print("Failed to initialize vector store. Exiting.")
                return
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Usage: python -m src.main [--clear|--refresh]")
            return
    else:
        # Load embeddings
        vector_store = load_embeddings()
        if vector_store is None:
            print("Failed to initialize vector store. Exiting.")
            return
    
    # Continue with query interface
    print("\nAurora Q&A System ready!")
    print("\n" + "=" * 80)
    print("Terminal Query Interface")
    print("=" * 80)
    print("Enter your questions below. Type 'quit' or 'exit' to stop.\n")
    
    # Interactive query loop
    while True:
        try:
            query = input("Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not query:
                continue
            
            print("\nSearching...")
            results = vector_store.search(query, top_k=5)
            display_search_results(results, query)
            print()  # Empty line for readability
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()

