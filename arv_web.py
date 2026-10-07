"""Adversarial Reviewer Agent — local web UI (FastAPI + SSE)."""
from __future__ import annotations

import asyncio
import json
import os
import queue
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

load_dotenv()

from arv_agent.graph import build_graph
from arv_agent.state import AgentState

app = FastAPI(title="Adversarial Reviewer Agent")

# job_id -> {"queue": queue.Queue, "report": str | None}
_jobs: dict[str, dict] = {}

_CHECK_LABELS = {
    "statistical":   "Statistical sanity",
    "claim_support": "Claim support",
    "citation":      "Citation quality",
    "language":      "Language red flags",
    "methodology":   "Methodology audit",
}


def _make_initial(source: str) -> AgentState:
    return {
        "paper_source": source,
        "paper_text": "",
        "paper_metadata": {},
        "claims": [],
        "paper_type": "unclear",
        "planned_checks": [],
        "checks_completed": [],
        "findings": [],
        "errors": [],
        "final_report": None,
        "status": "running",
    }


def _diff_states(prev: dict, curr: dict) -> list[dict]:
    events: list[dict] = []

    # Ingestion done
    if not prev.get("paper_metadata") and curr.get("paper_metadata"):
        meta = curr["paper_metadata"]
        authors = meta.get("authors", "")
        if isinstance(authors, list):
            authors = ", ".join(str(a) for a in authors)
        events.append({
            "type": "step",
            "step": "ingest",
            "title": (meta.get("title") or "untitled")[:80],
            "authors": str(authors),
            "year": str(meta.get("year", "")),
        })

    # Classification done (paper_type changed from "unclear")
    if (
        prev.get("paper_type") != curr.get("paper_type")
        and curr.get("paper_type") not in ("unclear", "")
    ):
        events.append({
            "type": "step",
            "step": "classify",
            "paper_type": curr["paper_type"],
        })

    # Planning done
    prev_planned = prev.get("planned_checks") or []
    curr_planned = curr.get("planned_checks") or []
    if not prev_planned and curr_planned:
        events.append({
            "type": "step",
            "step": "plan",
            "claims_n": len(curr.get("claims") or []),
            "checks": curr_planned,
            "paper_type": curr.get("paper_type", "unclear"),
        })

    # A check completed → emit new findings
    prev_done = prev.get("checks_completed") or []
    curr_done = curr.get("checks_completed") or []
    if len(curr_done) > len(prev_done):
        last_check = curr_done[-1]
        prev_findings = prev.get("findings") or []
        curr_findings = curr.get("findings") or []
        new_findings = curr_findings[len(prev_findings):]
        events.append({
            "type": "step",
            "step": "check",
            "check": last_check,
            "label": _CHECK_LABELS.get(last_check, last_check),
            "findings_total": len(curr_findings),
            "new_findings": new_findings,
        })

    return events


def _run_agent(job_id: str, source: str, tmp_path: Optional[str]) -> None:
    q = _jobs[job_id]["queue"]
    try:
        graph = build_graph()
        initial = _make_initial(source)
        prev = dict(initial)

        for state in graph.stream(initial, stream_mode="values"):
            for event in _diff_states(prev, state):
                q.put(event)
            prev = state

        report = prev.get("final_report") or "No report generated."
        _jobs[job_id]["report"] = report
        q.put({
            "type": "done",
            "report": report,
            "errors": prev.get("errors") or [],
        })
    except Exception as exc:
        q.put({"type": "error", "message": str(exc)})
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    path = Path(__file__).parent / "templates" / "arv.html"
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.post("/analyze")
async def analyze(
    paper_url: str = Form(default=""),
    file: UploadFile = File(default=None),
):
    job_id = str(uuid.uuid4())[:8]
    tmp_path: Optional[str] = None

    if file and file.filename:
        suffix = Path(file.filename).suffix or ".pdf"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(await file.read())
        tmp.close()
        source = tmp_path = tmp.name
    elif paper_url.strip():
        source = paper_url.strip()
    else:
        return {"error": "Provide a URL or upload a file."}

    _jobs[job_id] = {"queue": queue.Queue(), "report": None}
    threading.Thread(
        target=_run_agent, args=(job_id, source, tmp_path), daemon=True
    ).start()

    return {"job_id": job_id}


@app.get("/stream/{job_id}")
async def stream_events(job_id: str) -> StreamingResponse:
    if job_id not in _jobs:
        async def _missing():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Job not found'})}\n\n"
        return StreamingResponse(_missing(), media_type="text/event-stream")

    async def gen():
        q = _jobs[job_id]["queue"]
        while True:
            try:
                item = q.get_nowait()
                yield f"data: {json.dumps(item)}\n\n"
                if item["type"] in ("done", "error"):
                    break
            except queue.Empty:
                yield ": ping\n\n"
                await asyncio.sleep(0.3)
        _jobs.pop(job_id, None)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("web_app:app", host=host, port=port, reload=False)
