"""HTTP adapter for the original semantic retrieval and multi-turn logic."""
import asyncio
import hmac
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from . import core

logger = logging.getLogger(__name__)
@asynccontextmanager
async def lifespan(app):
    await run_in_threadpool(core.initialize)
    app.state.lock = asyncio.Lock()
    yield

app = FastAPI(title="창업 안내 RAG", lifespan=lifespan, docs_url=None, redoc_url=None)

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=6000)

class ChatInput(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    history: list[Message] = Field(default_factory=list, max_length=6)

    @field_validator("question")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("질문을 입력해 주세요.")
        return value.strip()

@app.get("/api/health")
async def health():
    return {"ready": core.retriever is not None,
            "mode": "generation" if core.OPENAI_API_KEY else "retrieval",
            "model": core.EMBEDDING_MODEL}

@app.post("/api/chat")
async def chat(payload: ChatInput, request: Request):
    # Optional shared access code is configured on the backend, never embedded in JS.
    expected = os.environ.get("DEMO_ACCESS_CODE", "")
    if expected and not hmac.compare_digest(request.headers.get("X-Demo-Code", ""), expected):
        raise HTTPException(401, "체험 코드를 확인해 주세요.")
    if core.OPENAI_API_KEY and not expected:
        raise HTTPException(503, "AI 답변 모드는 체험 코드 설정 후 사용할 수 있습니다.")
    if app.state.lock.locked():
        raise HTTPException(429, "다른 질문을 처리 중입니다. 잠시 후 다시 시도해 주세요.")
    async with app.state.lock:
        try:
            answer, hits = await run_in_threadpool(
                core.rag_answer, payload.question,
                history=[m.model_dump() for m in payload.history])
        except Exception:
            logger.exception("RAG request failed")
            raise HTTPException(503, "검색 서버를 잠시 사용할 수 없습니다.")
    seen, sources = set(), []
    for hit in hits:
        if hit["doc_id"] not in seen:
            sources.append({k: hit[k] for k in ["doc_id", "title", "category", "source", "score"]})
            seen.add(hit["doc_id"])
    return {"answer": answer, "sources": sources,
            "mode": "generation" if core.OPENAI_API_KEY else "retrieval"}

WEB = Path(__file__).resolve().parents[1] / "web"
app.mount("/assets", StaticFiles(directory=WEB), name="assets")
@app.get("/")
async def index():
    return FileResponse(WEB / "index.html")
