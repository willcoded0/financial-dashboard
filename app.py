"""
app.py - Flask web application for the Financial Dashboard.

Serves a file-upload interface, runs the ETL pipeline per session,
and proxies Ollama AI requests to the configured remote host.
"""

import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

import requests
import yaml
from flask import (Flask, Response, abort, redirect, render_template,
                   request, stream_with_context, url_for)
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"csv"}
MAX_UPLOAD_BYTES    = 50 * 1024 * 1024          # 50 MB total per request
SESSION_TTL_SECONDS = 24 * 3600                  # sessions live 24 hours
CLEANUP_INTERVAL    = 3600                        # check every hour

OLLAMA_HOST  = os.environ.get("OLLAMA_HOST",  "https://ai.batmap.win")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY") or os.urandom(32)


# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------

@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"]  = "nosniff"
    response.headers["X-Frame-Options"]          = "DENY"
    response.headers["X-XSS-Protection"]         = "1; mode=block"
    response.headers["Referrer-Policy"]           = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"]        = "geolocation=(), microphone=(), camera=()"
    # Allow Chart.js CDN and same-origin scripts only
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline'; "
        "connect-src 'self' https://ai.batmap.win; "
        "img-src 'self' data:; "
        "font-src 'self'; "
        "frame-ancestors 'none';"
    )
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def validate_session_id(session_id: str) -> bool:
    """Return True only if session_id is a valid UUID4 (prevents path traversal)."""
    try:
        val = uuid.UUID(session_id, version=4)
        return str(val) == session_id
    except ValueError:
        return False


def get_session_path(session_id: str) -> Path:
    """Return resolved session path, ensuring it stays inside SESSIONS_DIR."""
    base    = SESSIONS_DIR.resolve()
    target  = (SESSIONS_DIR / session_id).resolve()
    if not str(target).startswith(str(base)):
        abort(403)
    return target


# ---------------------------------------------------------------------------
# Background session cleanup
# ---------------------------------------------------------------------------

def _cleanup_loop():
    while True:
        time.sleep(CLEANUP_INTERVAL)
        now = time.time()
        try:
            for entry in SESSIONS_DIR.iterdir():
                if entry.is_dir():
                    age = now - entry.stat().st_mtime
                    if age > SESSION_TTL_SECONDS:
                        shutil.rmtree(entry, ignore_errors=True)
        except Exception:
            pass  # never crash the cleanup thread


_cleanup_thread = threading.Thread(target=_cleanup_loop, daemon=True)
_cleanup_thread.start()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    uploaded_files = request.files.getlist("csv_files")

    if not uploaded_files or all(f.filename == "" for f in uploaded_files):
        return render_template("index.html", error="Please select at least one CSV file.")

    # Create isolated session directories
    session_id = str(uuid.uuid4())
    session_dir = SESSIONS_DIR / session_id
    data_dir    = session_dir / "data"
    output_dir  = session_dir / "output"
    data_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    try:
        # Save uploaded files with sanitized names
        saved_count = 0
        for f in uploaded_files:
            if f and f.filename and allowed_file(f.filename):
                safe_name = secure_filename(f.filename)
                if not safe_name:
                    continue
                f.save(data_dir / safe_name)
                saved_count += 1

        if saved_count == 0:
            raise ValueError("No valid .csv files were uploaded.")

        # ── Run the ETL pipeline ────────────────────────────────────────────
        from src.ingest    import load_directory
        from src.transform import transform
        from src.analyze   import analyze
        from src.export    import export_csvs, generate_html_dashboard

        # 1. Ingest
        df = load_directory(data_dir)

        # 2. Transform
        df = transform(df)

        # 3. Optional date-range filter (applied after transform so year_month exists)
        start = request.form.get("start") or None
        end   = request.form.get("end")   or None
        if start:
            df = df[df["year_month"] >= start]
        if end:
            df = df[df["year_month"] <= end]

        if df.empty:
            raise ValueError("No transactions found in the selected date range.")

        # 4. Load budgets from config
        cat_path = Path(__file__).parent / "config" / "categories.yaml"
        budgets: dict = {}
        try:
            with open(cat_path) as fh:
                cfg = yaml.safe_load(fh)
            budgets = {k: float(v) for k, v in (cfg.get("budgets") or {}).items()}
        except Exception:
            pass

        # 5. Analyze
        balance = 0.0
        try:
            balance = float(request.form.get("balance") or "0")
        except ValueError:
            pass

        results = analyze(df, starting_balance=balance, budgets=budgets)

        # 6. Export CSVs + HTML
        export_csvs(results, output_dir)
        generate_html_dashboard(results, output_dir / "dashboard.html")

        # 7. Patch the generated HTML to use the server-side proxy
        #    so the AI chat works from any browser (no localhost required)
        dash_path = output_dir / "dashboard.html"
        html = dash_path.read_text(encoding="utf-8")

        html = html.replace(
            "http://localhost:11434/api/tags",
            "/api/ollama/tags",
        )
        html = html.replace(
            "http://localhost:11434/api/chat",
            "/api/ollama/chat",
        )
        html = html.replace(
            'model: "puck"',
            f'model: "{OLLAMA_MODEL}"',
        )
        html = html.replace(
            "⚠ Can't reach Ollama. Make sure it's running: ollama serve",
            "⚠ AI assistant is currently unavailable.",
        )

        dash_path.write_text(html, encoding="utf-8")

    except Exception as e:
        shutil.rmtree(session_dir, ignore_errors=True)
        return render_template("index.html", error=f"Processing failed: {e}")

    return redirect(url_for("dashboard", session_id=session_id))


@app.route("/dashboard/<session_id>")
def dashboard(session_id: str):
    if not validate_session_id(session_id):
        abort(404)

    dash_path = get_session_path(session_id) / "output" / "dashboard.html"
    if not dash_path.exists():
        abort(404)

    return Response(
        dash_path.read_text(encoding="utf-8"),
        mimetype="text/html",
    )


# ---------------------------------------------------------------------------
# Ollama proxy routes
# ---------------------------------------------------------------------------

@app.route("/api/ollama/tags")
def ollama_tags():
    """Health-check proxy — lets the dashboard verify the AI is reachable."""
    try:
        r = requests.get(
            f"{OLLAMA_HOST}/api/tags",
            timeout=5,
            headers={"Accept": "application/json"},
        )
        return Response(r.content, status=r.status_code,
                        content_type=r.headers.get("Content-Type", "application/json"))
    except Exception:
        return Response(
            json.dumps({"error": "AI service unavailable"}),
            status=503,
            content_type="application/json",
        )


@app.route("/api/ollama/chat", methods=["POST"])
def ollama_chat():
    """Streaming chat proxy — forwards requests to the hosted Ollama instance."""
    payload = request.get_json(silent=True)
    if not payload:
        return Response(
            json.dumps({"error": "Invalid request body"}),
            status=400,
            content_type="application/json",
        )

    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json=payload,
            stream=True,
            timeout=120,
            headers={"Content-Type": "application/json"},
        )

        def generate():
            for chunk in r.iter_content(chunk_size=None):
                if chunk:
                    yield chunk

        return Response(
            stream_with_context(generate()),
            status=r.status_code,
            content_type=r.headers.get("Content-Type", "application/json"),
        )
    except Exception as e:
        return Response(
            json.dumps({"error": str(e)}),
            status=503,
            content_type="application/json",
        )


# ---------------------------------------------------------------------------
# Error pages
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return render_template("index.html", error="Dashboard not found or has expired."), 404


@app.errorhandler(413)
def too_large(e):
    return render_template("index.html", error="Upload too large. Max 50 MB total."), 413


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    host  = os.environ.get("HOST",  "0.0.0.0")
    port  = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host=host, port=port, debug=debug)
