from pathlib import Path
from dotenv import load_dotenv
from src.vector_store.embedding_init import load_embeddings
from src.qa_engine import generate_answer, display_answer

env_path1 = Path(__file__).parent.parent / ".env"
load_dotenv(env_path1)

def main():
    """Main entry point with terminal query interface."""
    vector_store = load_embeddings()
    if vector_store is None:
        print("Failed to initialize vector store. Exiting.")
        return
    
    while True:
        try:
            query = input("Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            results = vector_store.search(query, top_k=5)
            answer = generate_answer(query, results)
            display_answer(answer, query, results)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()

