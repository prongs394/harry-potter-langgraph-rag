from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from ollama import Client

from retrieval import retrieve


# defining the shared state
class RAGState(TypedDict):
    question: str
    chunks: list[dict]
    answer: str


# cnnect to the local Ollama API
ollama_client = Client(host="http://localhost:11434")

# node 1: Retrieve relevant chunks
def retrieve_node(state: RAGState):
    chunks = retrieve(state["question"])

    # Diagnostic: use only the passage containing the correct evidence
    chunks = [chunk for chunk in chunks if chunk["book"] == 1 and chunk["chunk_index"] == 483]

    return {"chunks": chunks}


# generate the answer using Qwen
def generate_node(state: RAGState):
    context = "\n\n".join(
        f"[Passage {i}, Book {chunk['book']}]\n{chunk['text']}"
        for i, chunk in enumerate(state["chunks"], start=1)
    )

    response = ollama_client.chat(
        model="dolphin-llama3:latest",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer questions using only the provided passages. "
                    "First identify the most relevant passage and the exact evidence it provides. "
                    "Then give a concise answer supported by that evidence. "
                    "If no passage supports an answer, say 'I don't know'. "
                    "Never invent citations or information."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Passages:\n{context}\n\n"
                    f"Question: {state['question']}\n\n"
                    "Respond in this format:\n"
                    "Evidence: <short exact quotation from a passage>\n"
                    "Answer: <answer based on that quotation>"
                ),
            },
        ],
        options={"temperature": 0},
    )

    return {"answer": response.message.content}


# build the LangGraph workflow
builder = StateGraph(RAGState)

builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

graph = builder.compile()


# test the complete RAG pipeline
if __name__ == "__main__":
    question = input("Ask a Harry Potter question: ")

    result = graph.invoke({
        "question": question,
        "chunks": [],
        "answer": ""
    })

    print("\nAnswer:", result["answer"])