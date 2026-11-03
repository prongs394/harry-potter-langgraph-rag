
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from ollama import Client

from retrieval import retrieve


# defining the shared state
class RAGState(TypedDict):
    question: str
    chunks: list[dict]
    answer: str
    rewrite_count: int


# connect to the local ollama api
ollama_client = Client(host="http://localhost:11434")


# retrieve relevant chunks
def retrieve_node(state: RAGState):
    chunks = retrieve(state["question"])
    return {"chunks": chunks}


# generate the answer using the local llm
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
                    "Answer the question using only the provided passages. "
                    "If the passages do not contain enough information, say exactly 'I don't know'. "
                    "Do not invent information."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Passages:\n{context}\n\n"
                    f"Question: {state['question']}"
                ),
            },
        ],
        options={"temperature": 0},
    )

    return {"answer": response.message.content.strip()}


# check if the question needs rewriting
def check_answer(state: RAGState):
    if (
        "i don't know" in state["answer"].lower()
        and state["rewrite_count"] < 1
    ):
        return "rewrite"

    return "end"


# rewrite the question using ollama
def rewrite_node(state: RAGState):
    response = ollama_client.chat(
        model="dolphin-llama3:latest",
        messages=[
            {
                "role": "system",
                "content": (
                    "Rewrite the question to improve search in the Harry Potter books. "
                    "Keep the same meaning. Do not add new facts. "
                    "Return only the rewritten question."
                ),
            },
            {
                "role": "user",
                "content": state["question"],
            },
        ],
        options={"temperature": 0},
    )

    new_question = response.message.content.strip()
    print("rewritten question:", new_question)

    return {
        "question": new_question or state["question"],
        "rewrite_count": state["rewrite_count"] + 1,
    }


# build the langgraph workflow
builder = StateGraph(RAGState)

builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)
builder.add_node("rewrite", rewrite_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")

builder.add_conditional_edges(
    "generate",
    check_answer,
    {
        "rewrite": "rewrite",
        "end": END,
    },
)

builder.add_edge("rewrite", "retrieve")

graph = builder.compile()


# test the complete rag pipeline
if __name__ == "__main__":
    question = input("Ask a Harry Potter question: ")

    result = graph.invoke({
        "question": question,
        "chunks": [],
        "answer": "",
        "rewrite_count": 0,
    })

    print("\nAnswer:", result["answer"])
