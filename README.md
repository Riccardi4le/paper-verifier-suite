---
title: Paper Verifier Suite
emoji: 🔬
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: Three LLM agents that attack academic paper quality.
---

# Paper Verifier Suite

Three LangGraph agents that attack the quality of academic papers from different angles, served from a single web app.

**Live demo:** https://huggingface.co/spaces/Riccardi4le/paper-verifier-suite

| Agent | Path | What it does |
|-------|------|--------------|
| **Adversarial Reviewer** | `/arv` | A hostile peer reviewer: p-hacking, GRIM failures, unsupported claims, citation padding, methodology gaps |
| **Citation Genealogy** | `/cg` | Traces a claim back through its citation chain to the primary source and scores the distortion at each hop |
| **Cross-Paper Contradiction** | `/cpc` | Clusters claims across N papers and diagnoses *why* conflicting papers disagree |

Each agent was built as a standalone project; this repo packages them together. The standalone repos hold the full write-ups and CLIs:

- [Adversarial-Review-agent](https://github.com/Riccardi4le/Adversarial-Review-agent)
- [citation-genealogy-agent](https://github.com/Riccardi4le/citation-genealogy-agent)
- [cross-paper-contradiction-agent](https://github.com/Riccardi4le/cross-paper-contradiction-agent)

## Architecture

```
                    uvicorn app:app  (port 7860)
                              │
            ┌─────────────────┼──────────────────┐
            │                 │                  │
     /arv  FastAPI     /cg  FastAPI      /cpc  Flask (a2wsgi)
            │                 │                  │
   arv_agent (LangGraph) cg_agent (LangGraph) cpc_agent (LangGraph)
            │                 │                  │
            └──────── Groq API (gpt-oss) ────────┘
                                         + local MiniLM embeddings
```

Every agent keeps its own UI and API, mounted under its own prefix. Running them in one process keeps the suite inside a single free-tier Space.

## Models

All LLM calls go to [Groq](https://groq.com)'s free tier:

| Model | Used for |
|-------|----------|
| `openai/gpt-oss-120b` | ARV adversarial checks, CG citation verdicts, CPC diagnoses (default) |
| `openai/gpt-oss-20b` | ARV classification and claim extraction; selectable in CPC |

CPC clusters claims with `all-MiniLM-L6-v2` embeddings, computed locally in the container.

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env        # add your GROQ_API_KEY
uvicorn app:app --port 7860
```

Open http://localhost:7860.

## Deploy on Hugging Face Spaces

1. Create a Docker Space and push this repo to it.
2. In **Settings → Variables and secrets**, add the secret `GROQ_API_KEY` (free key at https://console.groq.com/keys).

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *(required)* | Groq API key |
| `GROQ_FAST_MODEL` | `openai/gpt-oss-20b` | ARV classification / extraction |
| `GROQ_DEEP_MODEL` | `openai/gpt-oss-120b` | ARV adversarial checks |
| `LLM_MODEL` | `openai/gpt-oss-120b` | CG citation analysis |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | CPC default model |
| `OPENALEX_EMAIL` | — | CG: OpenAlex polite pool |

## License

MIT
