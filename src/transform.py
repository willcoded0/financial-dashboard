"""
transform.py - Clean, categorize, and enrich transaction data.

Responsibilities:
- Normalize merchant names
- Categorize transactions against rules in categories.yaml
- Flag potential duplicate transactions
- Add derived time fields (day_of_week, month, is_weekend)
"""

import re
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml


CATEGORIES_FILE = Path(__file__).parent.parent / "config" / "categories.yaml"

# Patterns stripped from merchant names (card processor noise)
_NOISE_PATTERNS = [
    r"\s*#\d+",          # location numbers like "WALMART #1234"
    r"\s*\d{4,}",        # long digit sequences
    r"\s+\d{1,3}$",      # trailing short numbers
    r"\*+\S*",           # asterisk codes like SQ *COFFEEPLACE
    r"\s{2,}",           # multiple spaces → single
]

_NOISE_RE = re.compile("|".join(_NOISE_PATTERNS), re.IGNORECASE)


def _load_categories(path: Optional[Path] = None) -> dict[str, list[str]]:
    """Load category rules from YAML file.

    Args:
        path: Path to categories.yaml. Defaults to config/categories.yaml.

    Returns:
        Dict mapping category name to list of keyword strings.
    """
    path = path or CATEGORIES_FILE
    if not path.exists():
        raise FileNotFoundError(f"Categories config not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("categories", {})


def clean_merchant_name(name: str) -> str:
    """
    Normalize a raw merchant/description string.

    Steps:
    1. Strip leading/trailing whitespace
    2. Remove noise patterns (numbers, asterisks)
    3. Title-case the result

    Args:
        name: Raw description string from the bank CSV.

    Returns:
        Cleaned merchant name.
    """
    name = str(name).strip()
    name = _NOISE_RE.sub(" ", name)
    name = re.sub(r"\s{2,}", " ", name).strip()
    return name.title()


def _categorize_single(description: str, categories: dict[str, list[str]]) -> str:
    """Match a single description against category keyword rules.

    Matching is case-insensitive substring search.

    Args:
        description: Raw or cleaned transaction description.
        categories: Dict from _load_categories().

    Returns:
        Category name, or "Other" if no match found.
    """
    desc_lower = description.lower()
    for category, keywords in categories.items():
        for kw in keywords:
            if str(kw).lower() in desc_lower:
                return category
    return "Other"


def categorize(df: pd.DataFrame, categories_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Add 'category' column to the DataFrame based on description matching.

    Cash App Withdrawal (cash-out to bank) and Deposits (add-cash from bank card)
    are forced to Transfer regardless of description, since they are inter-account
    movements that will also appear in the linked bank account.

    Args:
        df: DataFrame with at least 'description' and 'bank_category' columns.
        categories_path: Optional path override for categories.yaml.

    Returns:
        DataFrame with 'category' column added.
    """
    cats = _load_categories(categories_path)
    df = df.copy()
    df["category"] = df["description"].apply(lambda d: _categorize_single(d, cats))

    # Force Transfer for Cash App inter-account movements
    cash_app_transfer_types = {"withdrawal", "deposits"}
    mask = df["bank_category"].str.strip().str.lower().isin(cash_app_transfer_types)
    df.loc[mask, "category"] = "Transfer"

    return df


def add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based derived fields to the DataFrame.

    Added columns:
        merchant       - cleaned version of description
        day_of_week    - Monday=0 ... Sunday=6
        day_name       - 'Monday', 'Tuesday', ...
        month          - integer month (1-12)
        month_name     - 'January', 'February', ...
        year           - integer year
        year_month     - string like '2024-03'
        is_weekend     - bool, True if Saturday or Sunday
        is_expense     - bool, True if amount < 0
        abs_amount     - absolute value of amount

    Args:
        df: DataFrame with 'date' and 'description' columns.

    Returns:
        Enriched DataFrame.
    """
    df = df.copy()
    # Ensure date column is always proper datetime (guards against mixed types from new CSVs)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    df["merchant"] = df["description"].apply(clean_merchant_name)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_name"] = df["date"].dt.day_name()
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%B")
    df["year"] = df["date"].dt.year
    df["year_month"] = df["date"].dt.strftime("%Y-%m")
    df["is_weekend"] = df["day_of_week"] >= 5
    df["is_expense"] = df["amount"] < 0
    df["abs_amount"] = df["amount"].abs()
    return df


def flag_duplicates(df: pd.DataFrame, window_days: int = 2) -> pd.DataFrame:
    """
    Flag rows that appear to be duplicates.

    A duplicate is defined as a row with the same (date ± window_days,
    amount, description) appearing more than once.

    Args:
        df: Transaction DataFrame with 'date', 'amount', 'description'.
        window_days: Number of days within which identical transactions
                     are considered potential duplicates.

    Returns:
        DataFrame with 'is_duplicate' boolean column added.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    df["is_duplicate"] = False

    for i, row in df.iterrows():
        if df.at[i, "is_duplicate"]:
            continue
        mask = (
            (df.index != i)
            & (df["amount"] == row["amount"])
            & (df["description"] == row["description"])
            & ((df["date"] - row["date"]).abs() <= pd.Timedelta(days=window_days))
        )
        dup_indices = df[mask].index
        if len(dup_indices) > 0:
            df.at[i, "is_duplicate"] = True
            df.loc[dup_indices, "is_duplicate"] = True

    return df


def transform(df: pd.DataFrame, categories_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Run the full transformation pipeline on ingested transaction data.

    Steps:
    1. Categorize transactions
    2. Add derived fields (merchant name, time fields, expense flags)
    3. Flag potential duplicates

    Args:
        df: Raw normalized DataFrame from ingest module.
        categories_path: Optional override for categories.yaml path.

    Returns:
        Fully transformed DataFrame ready for analysis.
    """
    print("Transforming transactions...")
    df = categorize(df, categories_path)
    df = add_derived_fields(df)
    df = flag_duplicates(df)

    dup_count = df["is_duplicate"].sum()
    if dup_count > 0:
        print(f"  Flagged {dup_count} potential duplicate transactions")

    cat_counts = df["category"].value_counts()
    print("  Category distribution:")
    for cat, count in cat_counts.items():
        print(f"    {cat}: {count}")

    return df
