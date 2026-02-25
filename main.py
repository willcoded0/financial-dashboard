"""
main.py - CLI entry point for the Financial Dashboard ETL pipeline.

Usage:
    python main.py --input data/ --output output/
    python main.py --input data/ --output output/ --start 2024-01 --end 2024-12
    python main.py --input data/ --output output/ --balance 5000
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="financial-dashboard",
        description="ETL pipeline for personal bank transaction analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input data/ --output output/
  python main.py --input data/ --output output/ --start 2024-01 --end 2024-12
  python main.py --input data/ --output output/ --balance 5000.00
        """,
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("data"),
        help="Directory containing bank CSV files (default: data/)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("output"),
        help="Directory for output files (default: output/)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        metavar="YYYY-MM",
        help="Filter transactions from this year-month (e.g. 2024-01)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        metavar="YYYY-MM",
        help="Filter transactions up to this year-month (e.g. 2024-12)",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=0.0,
        metavar="AMOUNT",
        help="Starting account balance for running balance calculation (default: 0)",
    )
    parser.add_argument(
        "--categories",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to custom categories.yaml (default: config/categories.yaml)",
    )
    parser.add_argument(
        "--std-threshold",
        type=float,
        default=2.0,
        metavar="FLOAT",
        help="Z-score threshold for anomaly detection (default: 2.0)",
    )
    return parser.parse_args()


def filter_by_date_range(
    df: pd.DataFrame,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    """Filter a DataFrame to the given year-month range (inclusive)."""
    if start:
        df = df[df["year_month"] >= start]
    if end:
        df = df[df["year_month"] <= end]
    return df


def main() -> int:
    args = parse_args()

    print("=" * 60)
    print("  Financial Dashboard ETL Pipeline")
    print("=" * 60)
    print(f"  Input:   {args.input.resolve()}")
    print(f"  Output:  {args.output.resolve()}")
    if args.start or args.end:
        print(f"  Range:   {args.start or 'beginning'} → {args.end or 'end'}")
    print()

    # --- Ingest ---
    from src.ingest import load_directory

    try:
        print("Step 1/4  Ingesting CSVs...")
        df = load_directory(args.input)
    except (FileNotFoundError, ValueError) as e:
        print(f"\nError during ingestion: {e}", file=sys.stderr)
        return 1

    if df.empty:
        print("No transactions found. Exiting.", file=sys.stderr)
        return 1

    # --- Transform ---
    from src.transform import transform

    print("\nStep 2/4  Transforming data...")
    try:
        df = transform(df, categories_path=args.categories)
    except FileNotFoundError as e:
        print(f"\nError loading categories: {e}", file=sys.stderr)
        return 1

    # --- Date range filter (applied after transform so year_month exists) ---
    if args.start or args.end:
        original_len = len(df)
        df = filter_by_date_range(df, args.start, args.end)
        print(f"  Date filter: {original_len} → {len(df)} transactions")
        if df.empty:
            print("No transactions in the specified date range.", file=sys.stderr)
            return 1

    # --- Analyze ---
    from src.analyze import analyze
    import yaml

    # Load optional budget limits from categories.yaml
    cat_path = args.categories or (Path(__file__).parent / "config" / "categories.yaml")
    budgets: dict[str, float] = {}
    try:
        with open(cat_path) as f:
            _cfg = yaml.safe_load(f)
        budgets = {k: float(v) for k, v in (_cfg.get("budgets") or {}).items()}
    except Exception:
        pass

    print("\nStep 3/4  Analyzing...")
    results = analyze(df, starting_balance=args.balance, std_threshold=args.std_threshold, budgets=budgets)

    # --- Export ---
    from src.export import export

    print("\nStep 4/4  Exporting...")
    try:
        export(results, args.output)
    except Exception as e:
        print(f"\nError during export: {e}", file=sys.stderr)
        return 1

    print("\n" + "=" * 60)
    print("  Done! Open output/dashboard.html in your browser.")
    print("  For Power BI: import output/transactions_clean.csv")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
