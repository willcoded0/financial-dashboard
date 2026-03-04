"""
app.py - Flask web application for the Financial Dashboard.

Serves a file-upload interface, runs the ETL pipeline per session,
proxies Ollama AI requests, and supports Plaid bank connection.
"""

import json
import os
import queue
import shutil
import threading
import time
import uuid
from datetime import date, timedelta
from pathlib import Path

import requests
import yaml
from flask import (Flask, Response, abort, redirect, render_template,
                   request, send_file, stream_with_context, url_for)
from werkzeug.utils import secure_filename

SAMPLE_CSV = Path(__file__).parent / "data" / "sample_transactions.csv"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS  = {"csv"}
MAX_UPLOAD_BYTES    = 50 * 1024 * 1024
SESSION_TTL_SECONDS = 24 * 3600
CLEANUP_INTERVAL    = 3600

# Per-session SSE progress queues  { session_id: Queue }
_progress_queues: dict = {}

OLLAMA_HOST  = os.environ.get("OLLAMA_HOST",  "https://ai.batmap.win")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

PLAID_CLIENT_ID = os.environ.get("PLAID_CLIENT_ID", "")
PLAID_SECRET    = os.environ.get("PLAID_SECRET",    "")
PLAID_ENV       = os.environ.get("PLAID_ENV",       "development")
PLAID_ENABLED   = bool(PLAID_CLIENT_ID and PLAID_SECRET)

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
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"]         = "DENY"
    response.headers["X-XSS-Protection"]        = "1; mode=block"
    response.headers["Referrer-Policy"]          = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"]       = "geolocation=(), microphone=(), camera=()"
    response.headers["Content-Security-Policy"]  = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdn.plaid.com; "
        "style-src 'self' 'unsafe-inline'; "
        "connect-src 'self' https://ai.batmap.win https://production.plaid.com "
        "https://development.plaid.com https://sandbox.plaid.com; "
        "img-src 'self' data: https://*.plaid.com; "
        "font-src 'self'; "
        "frame-src https://cdn.plaid.com; "
        "frame-ancestors 'none';"
    )
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_session_id(session_id: str) -> bool:
    try:
        val = uuid.UUID(session_id, version=4)
        return str(val) == session_id
    except ValueError:
        return False


def get_session_path(session_id: str) -> Path:
    base   = SESSIONS_DIR.resolve()
    target = (SESSIONS_DIR / session_id).resolve()
    if not str(target).startswith(str(base)):
        abort(403)
    return target


def _load_budgets() -> dict:
    cat_path = Path(__file__).parent / "config" / "categories.yaml"
    try:
        with open(cat_path) as fh:
            cfg = yaml.safe_load(fh)
        return {k: float(v) for k, v in (cfg.get("budgets") or {}).items()}
    except Exception:
        return {}


def _emit(session_id: str, event: str, data: str) -> None:
    q = _progress_queues.get(session_id)
    if q:
        q.put(f"event: {event}\ndata: {data}\n\n")


def _patch_dashboard_html(dash_path: Path) -> None:
    """Replace localhost Ollama references with the server-side proxy."""
    html = dash_path.read_text(encoding="utf-8")
    html = html.replace("http://localhost:11434/api/tags", "/api/ollama/tags")
    html = html.replace("http://localhost:11434/api/chat", "/api/ollama/chat")
    html = html.replace('model: "puck"', f'model: "{OLLAMA_MODEL}"')
    html = html.replace(
        "⚠ Can't reach Ollama. Make sure it's running: ollama serve",
        "⚠ AI assistant is currently unavailable.",
    )
    dash_path.write_text(html, encoding="utf-8")


def _run_pipeline(df, output_dir: Path, balance: float = 0.0,
                  start: str = None, end: str = None,
                  on_progress=None) -> None:
    """Transform, filter, analyze, and export a transaction DataFrame."""
    from src.transform import transform
    from src.analyze   import analyze
    from src.export    import export_csvs, generate_html_dashboard

    def emit(msg):
        if on_progress:
            on_progress(msg)

    emit("Categorizing transactions\u2026")
    df = transform(df)

    if start:
        df = df[df["year_month"] >= start]
    if end:
        df = df[df["year_month"] <= end]

    if df.empty:
        raise ValueError("No transactions found in the selected date range.")

    emit("Crunching the numbers\u2026")
    results = analyze(df, starting_balance=balance, budgets=_load_budgets())

    emit("Building your dashboard\u2026")
    export_csvs(results, output_dir)
    generate_html_dashboard(results, output_dir / "dashboard.html")
    _patch_dashboard_html(output_dir / "dashboard.html")


def _upload_thread(session_id: str, data_dir: Path, output_dir: Path,
                   balance: float, start, end) -> None:
    """Background worker: runs the full pipeline and emits SSE progress."""
    def emit(msg):
        _emit(session_id, "progress", msg)

    try:
        emit("Reading your files\u2026")
        from src.ingest import load_directory
        df = load_directory(data_dir)
        _run_pipeline(df, output_dir, balance=balance, start=start, end=end,
                      on_progress=emit)
        _emit(session_id, "done", f"/dashboard/{session_id}")
    except Exception as e:
        shutil.rmtree(output_dir.parent, ignore_errors=True)
        _emit(session_id, "error", str(e))
    finally:
        # Keep queue alive briefly so the SSE consumer can drain it
        threading.Timer(60, lambda: _progress_queues.pop(session_id, None)).start()


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
                    if now - entry.stat().st_mtime > SESSION_TTL_SECONDS:
                        shutil.rmtree(entry, ignore_errors=True)
        except Exception:
            pass


threading.Thread(target=_cleanup_loop, daemon=True).start()


# ---------------------------------------------------------------------------
# Routes — main
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", plaid_enabled=PLAID_ENABLED)


@app.route("/sample.csv")
def sample_csv():
    return send_file(SAMPLE_CSV, as_attachment=True,
                     download_name="sample_transactions.csv")


@app.route("/demo")
def demo():
    session_id  = str(uuid.uuid4())
    session_dir = SESSIONS_DIR / session_id
    data_dir    = session_dir / "data"
    output_dir  = session_dir / "output"
    data_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    try:
        shutil.copy(SAMPLE_CSV, data_dir / "sample_transactions.csv")
        from src.ingest import load_directory
        df = load_directory(data_dir)
        _run_pipeline(df, output_dir)
    except Exception as e:
        shutil.rmtree(session_dir, ignore_errors=True)
        return render_template("index.html", plaid_enabled=PLAID_ENABLED,
                               error=f"Demo failed: {e}")

    return redirect(url_for("dashboard", session_id=session_id))


@app.route("/upload", methods=["POST"])
def upload():
    uploaded_files = request.files.getlist("csv_files")
    if not uploaded_files or all(f.filename == "" for f in uploaded_files):
        return _json_resp({"error": "Please select at least one CSV file."}, 400)

    session_id  = str(uuid.uuid4())
    session_dir = SESSIONS_DIR / session_id
    data_dir    = session_dir / "data"
    output_dir  = session_dir / "output"
    data_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    saved_count = 0
    for f in uploaded_files:
        if f and f.filename and allowed_file(f.filename):
            safe_name = secure_filename(f.filename)
            if safe_name:
                f.save(data_dir / safe_name)
                saved_count += 1

    if saved_count == 0:
        shutil.rmtree(session_dir, ignore_errors=True)
        return _json_resp({"error": "No valid .csv files were uploaded."}, 400)

    balance = 0.0
    try:
        balance = float(request.form.get("balance") or "0")
    except ValueError:
        pass

    _progress_queues[session_id] = queue.Queue()
    threading.Thread(
        target=_upload_thread,
        args=(session_id, data_dir, output_dir),
        kwargs={
            "balance": balance,
            "start":   request.form.get("start") or None,
            "end":     request.form.get("end")   or None,
        },
        daemon=True,
    ).start()

    return _json_resp({"session_id": session_id})


@app.route("/progress/<session_id>")
def progress_stream(session_id: str):
    if not validate_session_id(session_id):
        abort(404)
    q = _progress_queues.get(session_id)
    if q is None:
        abort(404)

    def generate():
        while True:
            try:
                msg = q.get(timeout=60)
                yield msg
                if "event: done\n" in msg or "event: error\n" in msg:
                    break
            except queue.Empty:
                yield ": keepalive\n\n"

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/dashboard/<session_id>")
def dashboard(session_id: str):
    if not validate_session_id(session_id):
        abort(404)
    dash_path = get_session_path(session_id) / "output" / "dashboard.html"
    if not dash_path.exists():
        abort(404)
    return Response(dash_path.read_text(encoding="utf-8"), mimetype="text/html")


# ---------------------------------------------------------------------------
# Routes — Plaid
# ---------------------------------------------------------------------------

def _get_plaid_client():
    import plaid
    from plaid.api import plaid_api

    env_map = {
        "sandbox":     plaid.Environment.Sandbox,
        "development": plaid.Environment.Development,
        "production":  plaid.Environment.Production,
    }
    cfg = plaid.Configuration(
        host=env_map.get(PLAID_ENV, plaid.Environment.Development),
        api_key={"clientId": PLAID_CLIENT_ID, "secret": PLAID_SECRET},
    )
    return plaid_api.PlaidApi(plaid.ApiClient(cfg))


def _json_resp(data: dict, status: int = 200) -> Response:
    return Response(json.dumps(data), status=status,
                    content_type="application/json")


@app.route("/api/plaid/link_token", methods=["POST"])
def plaid_link_token():
    if not PLAID_ENABLED:
        return _json_resp({"error": "Plaid not configured on this server."}, 503)
    try:
        from plaid.model.link_token_create_request import LinkTokenCreateRequest
        from plaid.model.link_token_create_request_user import LinkTokenCreateRequestUser
        from plaid.model.products import Products
        from plaid.model.country_code import CountryCode

        req = LinkTokenCreateRequest(
            products=[Products("transactions")],
            client_name="Financial Dashboard",
            country_codes=[CountryCode("US")],
            language="en",
            user=LinkTokenCreateRequestUser(client_user_id=str(uuid.uuid4())),
        )
        resp = _get_plaid_client().link_token_create(req)
        return _json_resp({"link_token": resp["link_token"]})
    except Exception as e:
        return _json_resp({"error": str(e)}, 500)


@app.route("/api/plaid/connect", methods=["POST"])
def plaid_connect():
    """Exchange public token, fetch transactions, run pipeline, return dashboard URL."""
    if not PLAID_ENABLED:
        return _json_resp({"error": "Plaid not configured on this server."}, 503)

    data         = request.get_json(silent=True) or {}
    public_token = data.get("public_token")
    days_back    = min(int(data.get("days_back", 180)), 730)  # cap at 2 years

    if not public_token:
        return _json_resp({"error": "Missing public_token."}, 400)

    session_id  = str(uuid.uuid4())
    session_dir = SESSIONS_DIR / session_id
    output_dir  = session_dir / "output"
    output_dir.mkdir(parents=True)

    try:
        import pandas as pd
        import plaid as plaid_lib
        from plaid.model.item_public_token_exchange_request import (
            ItemPublicTokenExchangeRequest,
        )
        from plaid.model.transactions_get_request import TransactionsGetRequest
        from plaid.model.transactions_get_request_options import (
            TransactionsGetRequestOptions,
        )

        client = _get_plaid_client()

        # 1. Exchange public token → access token
        exchange_resp = client.item_public_token_exchange(
            ItemPublicTokenExchangeRequest(public_token=public_token)
        )
        access_token = exchange_resp["access_token"]

        # 2. Fetch all transactions (paginated, retry if data not ready yet)
        end_dt   = date.today()
        start_dt = end_dt - timedelta(days=days_back)
        all_txns = []
        retries  = 0

        while retries < 6:
            try:
                resp = client.transactions_get(
                    TransactionsGetRequest(
                        access_token=access_token,
                        start_date=start_dt,
                        end_date=end_dt,
                        options=TransactionsGetRequestOptions(
                            offset=len(all_txns)
                        ),
                    )
                )
                all_txns.extend(resp["transactions"])
                if len(all_txns) >= resp["total_transactions"]:
                    break
            except plaid_lib.ApiException as exc:
                body = json.loads(exc.body)
                if body.get("error_code") == "PRODUCT_NOT_READY":
                    time.sleep(3)
                    retries += 1
                    continue
                raise

        if not all_txns:
            raise ValueError("No transactions returned by Plaid for that date range.")

        # 3. Normalise to our DataFrame schema
        #    Plaid: positive amount = money OUT (debit)
        #    Ours:  negative amount = expense, positive = income
        rows = [
            {
                "date":          pd.to_datetime(t["date"]),
                "description":   t["name"],
                "amount":        -float(t["amount"]),
                "bank_category": (t.get("category") or [""])[0],
                "source_file":   "plaid",
            }
            for t in all_txns
            if not t.get("pending", False)
        ]

        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("All transactions are pending — try again shortly.")

        # 4. Run ETL pipeline
        _run_pipeline(df, output_dir)

    except Exception as e:
        shutil.rmtree(session_dir, ignore_errors=True)
        return _json_resp({"error": str(e)}, 500)

    return _json_resp({"redirect": url_for("dashboard", session_id=session_id)})


# ---------------------------------------------------------------------------
# Routes — Ollama proxy
# ---------------------------------------------------------------------------

@app.route("/api/ollama/tags")
def ollama_tags():
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5,
                         headers={"Accept": "application/json"})
        return Response(r.content, status=r.status_code,
                        content_type=r.headers.get("Content-Type", "application/json"))
    except Exception:
        return _json_resp({"error": "AI service unavailable"}, 503)


@app.route("/api/ollama/chat", methods=["POST"])
def ollama_chat():
    payload = request.get_json(silent=True)
    if not payload:
        return _json_resp({"error": "Invalid request body"}, 400)
    try:
        r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload,
                          stream=True, timeout=120,
                          headers={"Content-Type": "application/json"})

        def generate():
            for chunk in r.iter_content(chunk_size=None):
                if chunk:
                    yield chunk

        return Response(stream_with_context(generate()), status=r.status_code,
                        content_type=r.headers.get("Content-Type", "application/json"))
    except Exception as e:
        return _json_resp({"error": str(e)}, 503)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return render_template("index.html", plaid_enabled=PLAID_ENABLED,
                           error="Dashboard not found or has expired."), 404


@app.errorhandler(413)
def too_large(e):
    return render_template("index.html", plaid_enabled=PLAID_ENABLED,
                           error="Upload too large. Max 50 MB total."), 413


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    host  = os.environ.get("HOST",  "0.0.0.0")
    port  = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host=host, port=port, debug=debug)
