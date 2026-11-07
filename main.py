
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List

import search

app = FastAPI(title="UniGuide AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str
    history: List[Message] = Field(default_factory=list)


@app.get("/")
def home():
    return {
        "message": "Welcome to UniGuide AI!",
        "status": "Running"
    }


@app.post("/ask")
def ask_question(request: AskRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Please enter a question."
        )

    try:
        history = []
        pending_question = None

        for message in request.history:

            if message.role == "user":
                pending_question = message.content

            elif message.role in ("bot", "assistant") and pending_question:
                history.append({
                    "question": pending_question,
                    "answer": message.content
                })
                pending_question = None

        search.conversation_history = history[-10:]

        search_question = search.rewrite_followup_question(
            request.question
        )

        results = search.hybrid_search(search_question)

        answer = search.generate_answer(
            request.question,
            results
        )

        sources = []
        seen = set()

        for result in results:
            document = result.get("source", "BZU.pdf")
            page = result.get("page")

            key = (document, page)

            if key not in seen:
                seen.add(key)

                sources.append({
                    "document": document,
                    "page": page
                })

            if len(sources) >= 5:
                break

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        print("API Error:", str(e))

        raise HTTPException(
            status_code=500,
            detail="An error occurred while generating the answer."
        )
