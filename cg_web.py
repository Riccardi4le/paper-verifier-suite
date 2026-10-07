"""FastAPI web interface for the Citation Genealogy Agent."""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from pathlib import Path
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from cg_agent.graph import build_graph
from cg_agent.state import AgentState

app = FastAPI(title="Citation Genealogy Agent")

_jobs: dict[str, asyncio.Queue] = {}


class RunRequest(BaseModel):
    claim: str = Field(min_length=3, max_length=2000)
    source: str = Field(min_length=3, max_length=1000)
    max_depth: int = Field(default=5, ge=1, le=8)


def _copy_tree(tree: dict) -> dict:
    return {k: dict(v) for k, v in tree.items()}


def _run_agent(
    run_id: str,
    claim: str,
    source: str,
    max_depth: int,
    loop: asyncio.AbstractEventLoop,
) -> None:
    q = _jobs[run_id]

    def emit(event: dict) -> None:
        asyncio.run_coroutine_threadsafe(q.put(json.dumps(event)), loop)

    try:
        graph = build_graph()
        initial: AgentState = {
            "original_claim": claim,
            "source_ref": source,
            "max_depth": max_depth,
            "tree": {},
            "queue": [],
            "visited": [],
            "errors": [],
            "current_node_id": None,
            "final_report": None,
            "status": "running",
        }

        emit({"type": "status", "message": "Agent started — tracing citation chain…"})

        prev_tree: dict = {}
        prev_report: str | None = None

        for state in graph.stream(initial, stream_mode="values"):
            current_tree: dict = state.get("tree", {})

            for nid, node in current_tree.items():
                prev = prev_tree.get(nid, {})
                paper = node.get("paper")

                # Paper newly fetched
                if paper and not prev.get("paper"):
                    emit({
                        "type": "node_fetched",
                        "node_id": nid,
                        "title": paper.get("title") or "Unknown",
                        "year": paper.get("year"),
                        "depth": node.get("depth", 0),
                        "parent_id": node.get("parent_id"),
                    })

                # Verdict newly set
                prev_verdict = prev.get("verdict")
                curr_verdict = node.get("verdict")
                if curr_verdict and curr_verdict != prev_verdict:
                    emit({
                        "type": "node_analyzed",
                        "node_id": nid,
                        "title": (paper or {}).get("title") or "Unknown",
                        "year": (paper or {}).get("year"),
                        "verdict": curr_verdict,
                        "reasoning": node.get("reasoning"),
                        "is_primary": node.get("is_primary", False),
                        "depth": node.get("depth", 0),
                        "parent_id": node.get("parent_id"),
                        "children": node.get("children", []),
                    })

            report = state.get("final_report")
            if report and report != prev_report:
                prev_report = report
                emit({"type": "report", "content": report})

            prev_tree = _copy_tree(current_tree)

    except Exception as exc:
        emit({"type": "error", "message": str(exc)})
    finally:
        asyncio.run_coroutine_threadsafe(q.put(None), loop)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    path = Path(__file__).parent / "templates" / "cg.html"
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.post("/run")
async def start_run(req: RunRequest) -> dict:
    run_id = str(uuid.uuid4())
    loop = asyncio.get_running_loop()
    _jobs[run_id] = asyncio.Queue()
    threading.Thread(
        target=_run_agent,
        args=(run_id, req.claim, req.source, req.max_depth, loop),
        daemon=True,
    ).start()
    return {"run_id": run_id}


async def _event_stream(run_id: str) -> AsyncIterator[str]:
    if run_id not in _jobs:
        yield f'data: {json.dumps({"type": "error", "message": "Run not found"})}\n\n'
        return
    q = _jobs[run_id]
    while True:
        payload = await q.get()
        if payload is None:
            yield 'data: {"type":"done"}\n\n'
            break
        yield f"data: {payload}\n\n"
    _jobs.pop(run_id, None)


@app.get("/stream/{run_id}")
async def stream_events(run_id: str) -> StreamingResponse:
    return StreamingResponse(
        _event_stream(run_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    print("Starting Citation Genealogy Agent web interface…")
    print("Open http://127.0.0.1:8000 in your browser")
    uvicorn.run("webapp:app", host="127.0.0.1", port=8000, reload=False)
