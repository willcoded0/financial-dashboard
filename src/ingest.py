"""
ingest.py - Load and parse bank transaction CSVs.

Supports Chase, Bank of America, Wells Fargo, and generic CSV formats.
Auto-detects format based on column headers.
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd


# Known bank CSV header signatures
BANK_FORMATS = {
    "chase": {
        "required_cols": {"Transaction Date", "Post Date", "Description", "Category", "Type", "Amount"},
        "date_col": "Transaction Date",
        "desc_col": "Description",
        "amount_col": "Amount",
        "date_formats": ["%m/%d/%Y"],
    },
    "bofa": {
        "required_cols": {"Date", "Description", "Amount", "Running Bal."},
        "date_col": "Date",
        "desc_col": "Description",
        "amount_col": "Amount",
        "date_formats": ["%m/%d/%Y"],
    },
    "wells_fargo": {
        "required_cols": {"Date", "Amount", "* ", "* ", "Description"},
        # WF has unnamed cols; detect by positional count instead
        "date_col": 0,
        "desc_col": 4,
        "amount_col": 1,
        "date_formats": ["%m/%d/%Y"],
        "no_header": True,
    },
    "generic": {
        "required_cols": set(),
        "date_col": None,
        "desc_col": None,
        "amount_col": None,
        "date_formats": ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y"],
    },
}

DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y", "%Y/%m/%d"]


def _parse_amount(value: str | float) -> float:
    """Convert amount strings to float, handling parentheses for negatives."""
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace(",", "").replace("$", "").replace(" ", "")
    if s.startswith("(") and s.endswith(")"):
        return -float(s[1:-1])
    return float(s)


def _parse_date(value: str, fmt: Optional[str] = None) -> pd.Timestamp:
    """Parse a date string, trying multiple formats if needed."""
    if fmt:
        return pd.to_datetime(value, format=fmt)
    for f in DATE_FORMATS:
        try:
            return pd.to_datetime(value, format=f)
        except (ValueError, TypeError):
            continue
    return pd.to_datetime(value)


def _detect_format(df: pd.DataFrame) -> str:
    """Detect bank format from DataFrame columns."""
    cols = set(df.columns.str.strip())
    if {"Transaction Date", "Post Date", "Description", "Amount"}.issubset(cols):
        return "chase"
    if {"Date", "Description", "Amount", "Running Bal."}.issubset(cols):
        return "bofa"
    if {"Transaction Description", "Transaction Date", "Transaction Type", "Transaction Amount"}.issubset(cols):
        return "capital_one"
    if {"Transaction Type", "Net Amount", "Status", "Notes"}.issubset(cols):
        return "cash_app"
    # Wells Fargo has no header row (5 columns, col[4] is description)
    if df.shape[1] == 5 and all(str(c).isdigit() or str(c).startswith("0") for c in [0]):
        return "wells_fargo"
    return "generic"


def _normalize_chase(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a Chase CSV into the standard schema."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    out = pd.DataFrame()
    out["date"] = df["Transaction Date"].apply(_parse_date)
    out["description"] = df["Description"].astype(str).str.strip()
    out["amount"] = df["Amount"].apply(_parse_amount)
    out["bank_category"] = df.get("Category", pd.Series(dtype=str)).fillna("").astype(str).str.strip()
    out["source_file"] = ""
    return out


def _normalize_bofa(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a Bank of America CSV into the standard schema."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    out = pd.DataFrame()
    out["date"] = df["Date"].apply(_parse_date)
    out["description"] = df["Description"].astype(str).str.strip()
    out["amount"] = df["Amount"].apply(_parse_amount)
    out["bank_category"] = ""
    out["source_file"] = ""
    return out


def _normalize_wells_fargo(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a Wells Fargo no-header CSV into the standard schema."""
    raw = raw.copy()
    out = pd.DataFrame()
    out["date"] = raw.iloc[:, 0].apply(_parse_date)
    out["description"] = raw.iloc[:, 4].astype(str).str.strip()
    out["amount"] = raw.iloc[:, 1].apply(_parse_amount)
    out["bank_category"] = ""
    out["source_file"] = ""
    return out


def _normalize_capital_one(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a Capital One CSV into the standard schema.

    Capital One exports all amounts as positive with a Transaction Type column
    ('Debit' or 'Credit') indicating direction. Debits are negated.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    out = pd.DataFrame()
    out["date"] = df["Transaction Date"].apply(_parse_date)
    out["description"] = df["Transaction Description"].astype(str).str.strip()
    amounts = df["Transaction Amount"].apply(_parse_amount)
    out["amount"] = amounts.where(
        df["Transaction Type"].str.strip().str.lower() == "credit",
        -amounts,
    )
    out["bank_category"] = ""
    out["source_file"] = ""
    return out


def _normalize_cash_app(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a Cash App CSV export into the standard schema.

    Cash App exports:
    - Net Amount is already signed (negative = money out)
    - Transaction Type tells us the nature: Cash Card, P2P, Withdrawal, Deposits, etc.
    - FAILED rows are dropped
    - Withdrawal (cash-out to bank) and Deposits (add-cash from bank card) are
      flagged as transfers via bank_category so transform.py can force the category.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Drop failed and non-USD rows
    df = df[df["Status"].str.strip().str.upper() == "COMPLETE"].copy()

    # Parse the signed Net Amount (e.g. "-$167.76" or "$14.00")
    amounts = df["Net Amount"].apply(_parse_amount)

    # Build a human-readable description
    txn_type = df["Transaction Type"].str.strip()
    notes = df["Notes"].astype(str).str.strip()
    sender = df["Name of sender/receiver"].astype(str).str.strip()

    def _build_desc(row_type, row_notes, row_sender):
        if row_type == "P2P":
            parts = [p for p in [row_notes, row_sender] if p and p.lower() != "nan"]
            return " — ".join(parts) if parts else "Cash App P2P"
        if row_type == "Withdrawal":
            return "Cash App Cash Out"
        if row_type == "Deposits":
            return "Cash App Add Cash"
        return row_notes if row_notes and row_notes.lower() != "nan" else f"Cash App {row_type}"

    out = pd.DataFrame()
    out["date"] = df["Date"].apply(lambda d: pd.to_datetime(str(d).split()[0]))
    out["description"] = [
        _build_desc(t, n, s) for t, n, s in zip(txn_type, notes, sender)
    ]
    out["amount"] = amounts.values
    # Store the original transaction type so transform.py can force Transfer
    # for Withdrawal (cash-out) and Deposits (add-cash) rows
    out["bank_category"] = txn_type.values
    out["source_file"] = ""
    return out


def _normalize_generic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to normalize an unknown CSV by guessing column roles.
    Looks for columns whose names hint at date, description, amount.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    cols_lower = {c.lower(): c for c in df.columns}

    date_col = next((cols_lower[k] for k in cols_lower if "date" in k), None)
    desc_col = next(
        (cols_lower[k] for k in cols_lower if any(x in k for x in ("desc", "memo", "name", "merchant", "payee"))),
        None,
    )
    amount_col = next(
        (cols_lower[k] for k in cols_lower if any(x in k for x in ("amount", "debit", "credit", "sum", "total"))),
        None,
    )

    if not date_col or not desc_col or not amount_col:
        raise ValueError(
            f"Cannot auto-detect columns. Found: {list(df.columns)}. "
            "Please rename columns to include 'date', 'description'/'memo', and 'amount'."
        )

    out = pd.DataFrame()
    out["date"] = df[date_col].apply(_parse_date)
    out["description"] = df[desc_col].astype(str).str.strip()
    out["amount"] = df[amount_col].apply(_parse_amount)
    out["bank_category"] = ""
    out["source_file"] = ""
    return out


def load_csv(filepath: str | Path) -> pd.DataFrame:
    """
    Load a single bank CSV file and return a normalized DataFrame.

    Standard schema:
        date          - pd.Timestamp
        description   - str
        amount        - float (negative = debit/expense)
        bank_category - str (original bank label, if any)
        source_file   - str (basename of source file)

    Args:
        filepath: Path to the CSV file.

    Returns:
        Normalized DataFrame.

    Raises:
        ValueError: If the file cannot be parsed.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Try with header first
    try:
        raw = pd.read_csv(filepath, encoding="utf-8", dtype=str, skip_blank_lines=True)
    except UnicodeDecodeError:
        raw = pd.read_csv(filepath, encoding="latin-1", dtype=str, skip_blank_lines=True)

    # Strip BOM and whitespace from column names
    raw.columns = raw.columns.str.strip().str.lstrip("\ufeff")

    fmt = _detect_format(raw)

    if fmt == "chase":
        out = _normalize_chase(raw)
    elif fmt == "bofa":
        out = _normalize_bofa(raw)
    elif fmt == "capital_one":
        out = _normalize_capital_one(raw)
    elif fmt == "cash_app":
        out = _normalize_cash_app(raw)
    elif fmt == "wells_fargo":
        # Re-read without header
        raw_noh = pd.read_csv(filepath, header=None, dtype=str, skip_blank_lines=True)
        out = _normalize_wells_fargo(raw_noh)
    else:
        out = _normalize_generic(raw)

    out["source_file"] = filepath.name
    out = out.dropna(subset=["date", "amount"])
    out = out.reset_index(drop=True)
    return out


def load_directory(directory: str | Path, pattern: str = "*.csv") -> pd.DataFrame:
    """
    Load all CSV files from a directory and concatenate them.

    Args:
        directory: Path to the directory containing CSV files.
        pattern: Glob pattern for CSV files (default: "*.csv").

    Returns:
        Combined normalized DataFrame from all matching files.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If no CSV files are found.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    csv_files = sorted(directory.glob(pattern))
    if not csv_files:
        raise ValueError(f"No CSV files found in {directory} matching pattern '{pattern}'")

    frames = []
    errors = []
    for f in csv_files:
        try:
            df = load_csv(f)
            frames.append(df)
            print(f"  Loaded {f.name}: {len(df)} transactions")
        except Exception as e:
            errors.append(f"  Skipped {f.name}: {e}")

    if errors:
        print("\nWarnings during ingestion:")
        for err in errors:
            print(err)

    if not frames:
        raise ValueError("No valid CSV files could be loaded.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)
    print(f"\nTotal transactions loaded: {len(combined)}")
    return combined
