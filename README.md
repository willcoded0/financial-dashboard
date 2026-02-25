# Financial Dashboard

A local ETL pipeline that ingests personal bank CSV exports, categorizes transactions, and generates a self-contained HTML dashboard with charts, budget tracking, and an AI chat assistant.

---

## Features

- **Multi-source ingestion** — Capital One 360 Checking, 360 Performance Savings, and Cash App CSV exports
- **Auto-categorization** — keyword rules in `config/categories.yaml` matched case-insensitively against transaction descriptions; first match wins
- **Duplicate detection** — cross-account transfer flagging and repeated charge detection
- **Analysis suite**
  - Monthly income vs. expenses with savings rate
  - Spending by category (monthly + month-over-month comparison)
  - Top merchants by total spend
  - Spending by day of week
  - Anomaly detection (z-score per category)
  - Recurring transaction identification (subscriptions, bills)
  - Running account balance
- **Budget tracking** — set monthly limits in the YAML; live progress bars in-dashboard, editable without re-running the pipeline
- **Puck AI assistant** — chat panel powered by a local [Ollama](https://ollama.ai) model; queries are automatically augmented with relevant transaction rows
- **Self-contained output** — single `dashboard.html` with all charts and data embedded; no server required

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.ai) running locally *(optional — only needed for the Puck chat assistant)*

---

## Setup

```bash
git clone https://github.com/yourname/financial-dashboard.git
cd financial-dashboard
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Pull the AI model (optional):

```bash
ollama pull qwen2.5-coder:14b
```

---

## Usage

Drop your bank CSV exports into `data/` and run:

```bash
python main.py
```

Open `output/dashboard.html` in any browser when it finishes.

### Options

```
--input   PATH       Directory containing bank CSVs  (default: data/)
--output  PATH       Output directory                (default: output/)
--start   YYYY-MM    Filter transactions from this month
--end     YYYY-MM    Filter transactions up to this month
--balance AMOUNT     Starting balance for the running balance chart (default: 0)
--categories PATH    Path to a custom categories.yaml
--std-threshold N    Z-score threshold for anomaly detection (default: 2.0)
```

### Examples

```bash
# All time, default output
python main.py

# Specific year with known starting balance
python main.py --start 2024-01 --end 2024-12 --balance 3200.00

# Custom data location
python main.py --input ~/Downloads/bank-exports/ --output ~/Desktop/results/
```

---

## Supported CSV Formats

| Source | Notes |
|---|---|
| Capital One 360 Checking / Savings | Standard Capital One export format |
| Cash App | Full activity CSV export from the app |

Place any number of CSV files in `data/`. The ingestor detects the format automatically by inspecting column headers.

---

## Customizing Categories

Edit `config/categories.yaml`. Keywords are matched case-insensitively against the transaction description. First match wins; unmatched transactions are labeled `Other`.

```yaml
categories:
  Fast Food:
    - mcdonald
    - chick-fil-a
    - doordash

  My Category:
    - some merchant name
    - another keyword
```

Set monthly budget limits in the same file:

```yaml
budgets:
  Fast Food:    200
  Groceries:    350
  Gas:          150
  Subscriptions: 80
```

---

## Outputs

| File | Description |
|---|---|
| `output/dashboard.html` | Self-contained interactive dashboard |
| `output/transactions_clean.csv` | All transactions with category, flags, and running balance |
| `output/monthly_summary.csv` | Income, expenses, net, and savings rate per month |
| `output/monthly_by_category.csv` | Spending per category per month |
| `output/anomalies.csv` | Flagged anomaly transactions |
| `output/recurring.csv` | Detected recurring charges |
| `output/top_merchants.csv` | Top merchants ranked by total spend |

---

## Puck — AI Assistant

The dashboard includes a chat panel powered by a local Ollama model. Puck receives a financial context summary at startup and injects relevant transaction rows dynamically for each query — so it can answer specific questions like "how much did I spend on Steam last month?" accurately.

If Ollama is not running, the chat panel shows offline status and the rest of the dashboard works normally.

**Recommended model:**

```bash
ollama pull qwen2.5-coder:14b
```

**Opening as a local file (CORS):** Chrome blocks `localhost` fetches from `file://` URLs. Either configure Ollama origins:

```bash
# Add to /etc/systemd/system/ollama.service.d/override.conf
[Service]
Environment="OLLAMA_ORIGINS=*"
```

Or serve the output directory instead:

```bash
python -m http.server 8080 --directory output/
# then open http://localhost:8080
```

---

## Project Structure

```
financial-dashboard/
├── main.py                 # CLI entry point and pipeline orchestration
├── requirements.txt
├── config/
│   └── categories.yaml     # Categorization rules and budget limits
├── data/
│   ├── .gitkeep
│   └── sample_transactions.csv
└── src/
    ├── ingest.py           # CSV parsing and format detection
    ├── transform.py        # Categorization, deduplication, feature engineering
    ├── analyze.py          # Summaries, anomaly detection, recurring identification
    └── export.py           # HTML dashboard generation
```

---

## Privacy

All processing is local. No transaction data is sent anywhere. The `data/` and `output/` directories are excluded from git via `.gitignore`. The dashboard loads Chart.js from a CDN — for fully offline use, download it and update the `<script>` tag to point to the local file.
