import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# load models
embedding_model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5",
    device="cpu"
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cpu"
)

# connect to ChromaDB
client = chromadb.PersistentClient(
    path=r"D:\projects\langgraph_rag_harrypotter\chroma_db"
)

collection = client.get_collection("hp_bge_contextual")


def retrieve(question, top_k=5):

    # Embed the question
    query_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    # retrieve Top-100 from ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=100
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    # add neighboring chunks to each retrieved chunk
    contexts = []

    for document, metadata in zip(documents, metadatas):
        book = metadata["book"]
        chunk_index = metadata["chunk_index"]

        neighbor_ids = [
            f"book_{book}_chunk_{i}"
            for i in [chunk_index - 1, chunk_index + 1]
            if i >= 0
        ]

        neighbors = collection.get(ids=neighbor_ids)

        neighbor_texts = dict(
            zip(neighbors["ids"], neighbors["documents"])
        )

        previous = neighbor_texts.get(
            f"book_{book}_chunk_{chunk_index - 1}", ""
        )
        following = neighbor_texts.get(
            f"book_{book}_chunk_{chunk_index + 1}", ""
        )

        context = " ".join(
            part for part in [previous, document, following] if part
        )

        contexts.append(context)

    # cross-encoder scores all 100 candidates
    pairs = [
        #(question, context)
        #for context in contexts
        (question, document)
        for document in documents
    ]

    scores = reranker.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False
    )

    # sort by Cross-Encoder score and keep Top-k
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: float(scores[i]),
        reverse=True
    )[:top_k]

    return [
        {
            "text": contexts[i],
            "book": metadatas[i]["book"],
            "chunk_index": metadatas[i]["chunk_index"],
            "score": float(scores[i])
        }
        for i in ranked_indices
    ]


# Run directly to test
if __name__ == "__main__":
    question = input("Ask a Harry Potter question: ")

    results = retrieve(question)

    for i, result in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print("Book:", result["book"])
        print("Chunk:", result["chunk_index"])
        print("Reranker score:", round(result["score"], 3))
        print("Text:", result["text"])