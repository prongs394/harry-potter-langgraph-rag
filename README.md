
# Harry Potter RAG Service

A two-stage Retrieval-Augmented Generation (RAG) backend built with FastAPI, LangGraph, ChromaDB, and Ollama. The service retrieves relevant text from Harry Potter books and generates answers using a local LLM.

---

## How It Works

1. Step 1: Dense Retrieval
   - Encodes user queries using BAAI/bge-base-en-v1.5.
   - Searches for the top 100 relevant chunks stored in a local ChromaDB collection.
   - During indexing, embeddings are created using each chunk and its neighboring chunks.

2. Step 2: Cross-Encoder Reranking
   - Reranks the 100 retrieved chunks using cross-encoder/ms-marco-MiniLM-L-6-v2.
   - Selects the top 5 chunks based on their scores.

3. Step 3: Context Window Expansion
   - Merges neighboring chunks (i-1, i, i+1) to provide more context for each selected chunk.

4. Step 4: Answer Generation
   - Sends the top 5 expanded passages to a local LLM through Ollama.
   - Generates answers with temperature 0.
   - Instructs the model to use only the retrieved passages.

5. Step 5: LangGraph Workflow
   - Uses LangGraph to manage retrieval, answer generation, and question rewriting.
   - Stores the question, retrieved passages, answer, and rewrite count in a shared state.
   - If the LLM responds with "I don't know", it rewrites the question using Ollama and runs retrieval and generation again.
   - Limits rewriting to one attempt to avoid an infinite loop.

---

## Tech Stack

- API Framework: FastAPI, Uvicorn, Pydantic
- Workflow: LangGraph
- Vector Database: ChromaDB (Persistent)
- Embedding Model: BAAI/bge-base-en-v1.5
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
- LLM Engine: Ollama (dolphin-llama3:latest)
- Document Processing: PyPDF

---

## Project Structure

- server.py: FastAPI application and API endpoint
- agent.py: LangGraph workflow, question rewriting, and answer generation
- retrieval.py: Dense retrieval, reranking, and context expansion
- index_books.ipynb: PDF processing, chunking, and vector database preparation
- requirements.txt: Python dependencies
- .gitignore: Excludes local database and temporary files
- README.md: Documentation

---

## Setup & Running Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/prongs394/harry-potter-langgraph-rag.git
   cd harry-potter-langgraph-rag
   ```

2. Install dependencies:

   Python 3.13 was used during development.

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Prepare Data & Build Vector Database:
   - Provide your own Harry Potter book files in PDF format. (No, the copyrighted books aren't in this repo. Accio Book won't fetch them for you either.)
   - Set the book file paths in `index_books.ipynb` and run it once to generate embeddings and build the local `chroma_db/` directory.
   - The database is not included in this repository.

4. Set Up Ollama:
   - Install Ollama from https://ollama.com/.
   - Download the local model:

   ```bash
   ollama pull dolphin-llama3:latest
   ```

   - Make sure Ollama is running at http://localhost:11434.

5. Start the service:

   ```bash
   python -m uvicorn server:app --host 127.0.0.1 --port 8000
   ```

   To access the API from another device on the same Wi-Fi, use `--host 0.0.0.0` and open `http://YOUR_LOCAL_IP:8000/docs` on that device.

---

## API Usage

- Interactive API Docs (Swagger UI):
  Visit http://localhost:8000/docs in your browser.

- Endpoint: POST /chat

- Sample cURL Request:

  ```bash
  curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d "{\"question\": \"What is the name of Harry Potter's owl?\"}"
  ```

- Sample Response:

  ```json
  {
    "answer": "The name of Harry Potter's owl is Hedwig."
  }
  ```

---

## Limitations

- The local LLM may give incorrect answers even when the correct information is retrieved.
- Question rewriting is only triggered when the LLM responds with "I don't know". It cannot detect every incorrect answer.
- The vector database must be built locally before running the service.
- Answer speed depends on the available hardware.
