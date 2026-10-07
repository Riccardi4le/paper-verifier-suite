import os
import uuid
import threading
import tempfile
import shutil
from pathlib import Path
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
BASE_DIR = Path(__file__).parent

PHASES = ["ingest", "extract", "cluster", "detect", "investigate", "build_map"]
PHASE_LABELS = {
    "ingest": "Reading Papers",
    "extract": "Extracting Key Claims",
    "cluster": "Grouping by Topic",
    "detect": "Detecting Conflicts",
    "investigate": "Deep Investigation",
    "build_map": "Building Report",
}

# job_id → {status, phase, phase_index, phase_label, result_markdown, stats, job_errors, error_message}
jobs: dict = {}

# Ensures only one job runs at a time (model patching is not thread-safe)
_global_lock = threading.Lock()


def _check_groq_ready(model: str = "") -> list[str]:
    """Returns the curated list of Groq models. Raises on missing key or invalid model."""
    from cpc_agent.config import GROQ_API_KEY, GROQ_AVAILABLE_MODELS

    if not GROQ_API_KEY:
        raise ConnectionError(
            "GROQ_API_KEY is not set. Add it to your .env file.\n"
            "Get a free key at https://console.groq.com/keys"
        )

    if model and model not in GROQ_AVAILABLE_MODELS:
        raise ValueError(
            f"Model '{model}' is not in the allowed list.\n"
            f"Allowed: {', '.join(GROQ_AVAILABLE_MODELS)}"
        )
    return list(GROQ_AVAILABLE_MODELS)


def _run_job(job_id: str, pdf_paths: list, model: str, topic: str, temp_dir: str):
    """Background worker — runs inside _global_lock so model patching is safe."""
    with _global_lock:
        job = jobs[job_id]
        try:
            job["status"] = "running"
            job["phase_label"] = "Connecting to Groq..."

            _check_groq_ready(model)

            # Patch module-level GROQ_MODEL so nodes pick up the chosen model.
            # Safe because _global_lock serialises all jobs.
            import cpc_agent.llm as _llm
            _llm.GROQ_MODEL = model
            _llm._client = None  # force client refresh

            from cpc_agent.graph import build_graph

            topic_str = topic or (
                ", ".join(Path(p).stem[:20] for p in pdf_paths[:3])
                + (" ..." if len(pdf_paths) > 3 else "")
            )

            initial_state = {
                "pdf_paths": pdf_paths,
                "topic": topic_str,
                "papers": [],
                "all_claims": [],
                "claims_per_paper": {},
                "clusters": [],
                "conflict_verdicts": {},
                "contradictions": [],
                "output_path": "",
                "errors": [],
            }

            graph = build_graph()
            accumulated = dict(initial_state)

            for event in graph.stream(initial_state):
                node_name = list(event.keys())[0]
                if node_name.startswith("__"):
                    continue
                state_update = list(event.values())[0]
                if isinstance(state_update, dict):
                    accumulated.update(state_update)

                if node_name in PHASES:
                    idx = PHASES.index(node_name)
                    job["phase"] = node_name
                    job["phase_index"] = idx
                    job["phase_label"] = PHASE_LABELS.get(node_name, node_name)

            output_path = accumulated.get("output_path", "")
            if output_path and Path(output_path).exists():
                job["result_markdown"] = Path(output_path).read_text(encoding="utf-8")
            else:
                job["result_markdown"] = "# No output generated\n\nCheck errors below."

            job["stats"] = {
                "papers": len(accumulated.get("papers", [])),
                "claims": len(accumulated.get("all_claims", [])),
                "clusters": len(accumulated.get("clusters", [])),
                "contradictions": len(accumulated.get("contradictions", [])),
            }
            job["job_errors"] = accumulated.get("errors", [])
            job["status"] = "done"

        except Exception as e:
            job["status"] = "error"
            job["error_message"] = str(e)
        finally:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("cpc.html")


@app.route("/api/models")
def api_models():
    try:
        available = _check_groq_ready()
        return jsonify({"models": available, "status": "ok"})
    except Exception as e:
        return jsonify({"models": [], "status": "error", "message": str(e)})


@app.route("/api/run", methods=["POST"])
def api_run():
    from cpc_agent.config import GROQ_MODEL

    files = request.files.getlist("papers")
    model = (request.form.get("model") or GROQ_MODEL).strip()
    topic = (request.form.get("topic") or "").strip()

    pdf_files = [f for f in files if f.filename.lower().endswith(".pdf")]
    if len(pdf_files) < 2:
        return jsonify({"error": "Upload at least 2 PDF files."}), 400

    temp_dir = tempfile.mkdtemp(prefix="cpc_")
    pdf_paths = []
    for f in pdf_files:
        safe_name = Path(f.filename).name
        dest = Path(temp_dir) / f"{uuid.uuid4().hex[:8]}_{safe_name}"
        f.save(str(dest))
        pdf_paths.append(str(dest))

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "queued",
        "phase": None,
        "phase_index": -1,
        "phase_label": "Queued...",
        "result_markdown": None,
        "stats": None,
        "job_errors": [],
        "error_message": None,
    }

    t = threading.Thread(
        target=_run_job,
        args=(job_id, pdf_paths, model, topic, temp_dir),
        daemon=True,
    )
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def api_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "status": job["status"],
        "phase": job["phase"],
        "phase_index": job["phase_index"],
        "phase_label": job["phase_label"],
        "error_message": job.get("error_message"),
    })


@app.route("/api/result/<job_id>")
def api_result(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] != "done":
        return jsonify({"error": "Job not finished yet"}), 400
    return jsonify({
        "markdown": job["result_markdown"],
        "stats": job["stats"],
        "errors": job["job_errors"],
    })


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    print("\n" + "=" * 50)
    print("  CPC Agent — Web UI")
    print(f"  Listening on http://{host}:{port}")
    print("  Prerequisite: GROQ_API_KEY env var (or .env)")
    print("=" * 50 + "\n")
    app.run(host=host, debug=False, port=port, threaded=True)
