"""Paper Verifier Suite — three paper-quality agents behind a single web app.

Each agent keeps its own UI and API, mounted under its own prefix:
  /arv  Adversarial Reviewer       (FastAPI)
  /cg   Citation Genealogy         (FastAPI)
  /cpc  Cross-Paper Contradiction  (Flask, via WSGI)

One process means one Hugging Face Space, which keeps the suite within the
free tier's limit of concurrently running Spaces.
"""
from __future__ import annotations

from pathlib import Path

from a2wsgi import WSGIMiddleware
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse

load_dotenv()

import arv_web  # noqa: E402
import cg_web  # noqa: E402
import cpc_web  # noqa: E402

app = FastAPI(title="Paper Verifier Suite", docs_url=None, redoc_url=None)

_INDEX = Path(__file__).parent / "templates" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(_INDEX.read_text(encoding="utf-8"))


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


# The agent UIs use relative URLs, so they must be served with a trailing slash.
def _add_slash_redirect(prefix: str) -> None:
    @app.get(f"/{prefix}", include_in_schema=False)
    async def _redirect() -> RedirectResponse:
        return RedirectResponse(f"{prefix}/")


for _prefix in ("arv", "cg", "cpc"):
    _add_slash_redirect(_prefix)

app.mount("/arv", arv_web.app)
app.mount("/cg", cg_web.app)
app.mount("/cpc", WSGIMiddleware(cpc_web.app))
