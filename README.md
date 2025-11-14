# Aurora Nov 7 - Romail Khan

Aurora Q/A System

Deployed with GCR

Docs: https://aurora-nov7-239357442081.us-central1.run.app/docs

Use the bot: https://aurora-nov7-239357442081.us-central1.run.app/ask

Query by sending a POST request to the /ask endpoint with a JSON body containing the question.
```json
{
    "query": "When is Layla planning her trip to London?"
}
```

Response:
```json
{
    "answer": "..."
}
```

## Demo
YT Link: https://youtu.be/yXsJaeIiY4I

<video src="https://youtu.be/yXsJaeIiY4I" width="320" height="240" controls></video>



**Embedding Strategy:**
- All member messages are embedded using OpenAI's `text-embedding-3-large` model
- Embeddings are stored in ChromaDB for semantic search
- On startup, embeddings are loaded from persistent storage or generated if missing (calls the API and fetches data)
- Query embeddings use the same model for consistent similarity matching

**RAG Pipeline:**
1. User query is embedded into the same vector space
2. ChromaDB performs semantic search to find top 5 most relevant messages
3. Retrieved messages are used as context for GPT-4o-mini to generate answers
4. Answers are based solely on the retrieved context

## Alternative Approaches

### Approach 1: PostgreSQL with pgvector


- Use PostgreSQL with pgvector extension
- Store embeddings as vector columns in SQL tables

**Embedding Strategy:**
- OpenAI `text-embedding-3-large` model
- Store vectors in PostgreSQL using `vector` data type
- Use HNSW index for fast similarity search

**RAG Pipeline:**
1. Query embedding generated using same model
2. PostgreSQL performs vector similarity search with SQL
3. Can combine semantic search with SQL WHERE clauses for filtering
4. Retrieved messages passed to LLM for answer generation

---

### Approach 2: Hybrid Search (Semantic + Keyword)

- Combine semantic search with traditional keyword/BM25 search
- Use weighted combination of both search results

**Embedding Strategy:**
- Same embedding model for semantic component
- Add BM25/keyword search index alongside vector store
- Weight results: 70% semantic, 30% keyword

**RAG Pipeline:**
1. Query processed through both semantic and keyword search
2. Results merged and reranked by relevance score
3. Top K results from combined search
4. Context passed to LLM for generation

---

### Approach 3: Local Embeddings with Sentence Transformers

- Replace OpenAI embeddings with local models such as sentence-transformers

**Embedding Strategy:**
- Use `all-MiniLM-L6-v2` or `sentence-transformers/all-mpnet-base-v2`
- Generate embeddings locally without API calls
- Store in ChromaDB

**RAG Pipeline:**
1. Query embedded using same local model
2. Vector search in ChromaDB
3. Top K results retrieved
4. GPT-4o-mini still used for answer generation

---

### Approach 4: Fine-tuned Domain-Specific Model

- Fine tune a small LLM on member messages

**Embedding Strategy:**
- Still use embeddings for initial retrieval
- Fine tuned model can answer directly from its training
- Use RAG for specific queries, direct generation for general ones

**RAG Pipeline:**
1. Classify query as general or specific
2. General queries: fine-tuned model direct answer
3. Specific queries: RAG pipeline (embed -> retrieve -> generate)

---

### Approach 5: Graph Database with LLM

- Store messages in a graph database like neo4j
- Model relationships between members, topics, conversations
- Use graph queries + LLM for complex relationship questions

**Embedding Strategy:**
- Embeddings stored as node properties
- Graph structure captures member interactions and topic connections
- Combine graph traversal with vector similarity

**RAG Pipeline:**
1. Query analyzed for relationship patterns
2. Graph query finds related nodes (members, topics, conversations)
3. Vector search on node embeddings for semantic similarity
4. Combined graph + vector results passed to LLM
5. LLM generates answer using relationship context
