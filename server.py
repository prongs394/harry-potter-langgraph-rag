from fastapi import FastAPI
from pydantic import BaseModel, Field

from agent import graph


# Create the FastAPI application
app = FastAPI(title="Harry Potter RAG API")


# Define the expected request body
class ChatRequest(BaseModel):
    question: str = Field(min_length=1)


# Define the response format
class ChatResponse(BaseModel):
    answer: str


# Create the chat endpoint
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = graph.invoke({
        "question": request.question,
        "chunks": [],
        "answer": ""
    })

    return ChatResponse(answer=result["answer"])