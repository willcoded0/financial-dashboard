"""
analyze.py - Calculate summaries and flag anomalies in transaction data.

Produces:
- Monthly spending by category
- Monthly income vs. expenses
- Anomaly detection (transactions > 2 std dev from category mean)
- Running balance
- Recurring transaction identification
"""

from typing import Optional

import numpy as np
import pandas as pd


def monthly_spending_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize expense totals by year-month and category.

    Excludes Transfer transactions (inter-account movements).

    Args:
        df: Transformed transaction DataFrame.

    Returns:
        DataFrame with columns: year_month, category, total_spent.
        total_spent is always positive (absolute value of expenses).
    """
    expenses = df[
        df["is_expense"] & ~df["is_duplicate"] & ~df["category"].isin({"Transfer"})
    ].copy()
    summary = (
        expenses.groupby(["year_month", "category"])["abs_amount"]
        .sum()
        .reset_index()
        .rename(columns={"abs_amount": "total_spent"})
    )
    summary = summary.sort_values(["year_month", "total_spent"], ascending=[True, False])
    return summary


def monthly_income_vs_expenses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize total income and total expenses per year-month.

    Excludes Transfer transactions (inter-account movements) from both sides
    so savings↔checking transfers and Cash App cash-outs don't inflate numbers.

    Args:
        df: Transformed transaction DataFrame.

    Returns:
        DataFrame with columns: year_month, income, expenses, net.
    """
    exclude_cats = {"Transfer"}
    # Only count the "Income" category as real income (payroll, interest, etc.)
    # P2P received money from friends/family lives in other categories and
    # is excluded here to avoid inflating the income figure.
    income = (
        df[df["category"] == "Income"]
        .groupby("year_month")["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "income"})
    )
    expense = (
        df[
            df["is_expense"]
            & ~df["is_duplicate"]
            & ~df["category"].isin(exclude_cats)
        ]
        .groupby("year_month")["abs_amount"]
        .sum()
        .reset_index()
        .rename(columns={"abs_amount": "expenses"})
    )
    summary = pd.merge(income, expense, on="year_month", how="outer").fillna(0)
    summary["net"] = summary["income"] - summary["expenses"]
    summary = summary.sort_values("year_month")
    return summary


def flag_anomalies(df: pd.DataFrame, std_threshold: float = 2.0) -> pd.DataFrame:
    """
    Flag transactions that are statistical anomalies within their category.

    A transaction is flagged when its absolute amount is more than
    ``std_threshold`` standard deviations above the category mean.

    Args:
        df: Transformed transaction DataFrame.
        std_threshold: Z-score threshold for flagging (default: 2.0).

    Returns:
        DataFrame with 'is_anomaly' and 'anomaly_zscore' columns added.
    """
    df = df.copy()
    df["is_anomaly"] = False
    zscore_map: dict[int, float] = {}

    exclude_from_anomaly = {"Transfer"}
    for category, group in df[df["is_expense"] & ~df["is_duplicate"]].groupby("category"):
        if category in exclude_from_anomaly:
            continue
        if len(group) < 3:
            continue
        mean = group["abs_amount"].mean()
        std = group["abs_amount"].std()
        if std == 0:
            continue
        zscores = (group["abs_amount"] - mean) / std
        for idx, z in zscores.items():
            zscore_map[idx] = round(float(z), 2)
            if z > std_threshold:
                df.at[idx, "is_anomaly"] = True

    df["anomaly_zscore"] = pd.array(
        [zscore_map.get(i, float("nan")) for i in df.index], dtype="Float64"
    )
    return df


def running_balance(df: pd.DataFrame, starting_balance: float = 0.0) -> pd.DataFrame:
    """
    Calculate a running balance column ordered by date.

    Args:
        df: Transaction DataFrame with 'date' and 'amount' columns.
        starting_balance: Initial account balance before first transaction.

    Returns:
        DataFrame sorted by date with 'running_balance' column added.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    df["running_balance"] = starting_balance + df["amount"].cumsum()
    return df


def find_recurring(
    df: pd.DataFrame,
    min_occurrences: int = 2,
    tolerance_days: int = 5,
    amount_tolerance: float = 1.0,
) -> pd.DataFrame:
    """
    Identify likely recurring transactions (subscriptions, bills).

    A transaction is considered recurring if:
    - The same merchant appears at least ``min_occurrences`` times
    - The intervals between occurrences cluster around 7, 14, or 30 days
    - The amounts are within ``amount_tolerance`` of each other

    Args:
        df: Transformed transaction DataFrame.
        min_occurrences: Minimum number of times merchant must appear.
        tolerance_days: Allowed deviation from expected interval (days).
        amount_tolerance: Maximum allowed dollar difference between amounts.

    Returns:
        DataFrame with columns: merchant, category, avg_amount,
        occurrences, interval_days (approximate).
    """
    expenses = df[df["is_expense"]].copy()
    recurring_rows = []

    for merchant, group in expenses.groupby("merchant"):
        if len(group) < min_occurrences:
            continue

        group = group.sort_values("date")
        amounts = group["abs_amount"].values
        dates = group["date"].values

        # Check if amounts are roughly consistent
        if amounts.max() - amounts.min() > amount_tolerance:
            continue

        # Calculate day intervals between consecutive occurrences
        if len(dates) < 2:
            continue
        intervals = [
            (pd.Timestamp(dates[i + 1]) - pd.Timestamp(dates[i])).days
            for i in range(len(dates) - 1)
        ]
        avg_interval = np.mean(intervals)

        # Accept if interval clusters around weekly, bi-weekly, or monthly
        is_recurring = any(
            abs(avg_interval - target) <= tolerance_days
            for target in [7, 14, 30, 31, 28, 29]
        )
        if not is_recurring:
            continue

        recurring_rows.append(
            {
                "merchant": merchant,
                "category": group["category"].mode().iloc[0],
                "avg_amount": round(float(amounts.mean()), 2),
                "occurrences": len(group),
                "interval_days": round(avg_interval, 1),
            }
        )

    if not recurring_rows:
        return pd.DataFrame(columns=["merchant", "category", "avg_amount", "occurrences", "interval_days"])

    result = pd.DataFrame(recurring_rows).sort_values("avg_amount", ascending=False)
    return result.reset_index(drop=True)


def top_merchants(df: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    """Rank merchants by total spend (expenses only, no transfers, no dupes)."""
    expenses = df[
        df["is_expense"] & ~df["is_duplicate"] & ~df["category"].isin({"Transfer"})
    ]
    if expenses.empty:
        return pd.DataFrame(columns=["merchant", "total_spent", "transactions", "avg_amount", "category"])
    result = (
        expenses.groupby("merchant")
        .agg(
            total_spent=("abs_amount", "sum"),
            transactions=("abs_amount", "count"),
            avg_amount=("abs_amount", "mean"),
            category=("category", lambda x: x.mode().iloc[0]),
        )
        .reset_index()
        .sort_values("total_spent", ascending=False)
        .head(n)
    )
    result["total_spent"] = result["total_spent"].round(2)
    result["avg_amount"] = result["avg_amount"].round(2)
    return result.reset_index(drop=True)


def spending_by_dow(df: pd.DataFrame) -> pd.DataFrame:
    """Total and average spend broken down by day of week."""
    expenses = df[
        df["is_expense"] & ~df["is_duplicate"] & ~df["category"].isin({"Transfer"})
    ]
    if expenses.empty:
        return pd.DataFrame()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    result = (
        expenses.groupby("day_name")
        .agg(total_spent=("abs_amount", "sum"), avg_per_txn=("abs_amount", "mean"), transactions=("abs_amount", "count"))
        .reset_index()
    )
    result["day_name"] = pd.Categorical(result["day_name"], categories=day_order, ordered=True)
    result = result.sort_values("day_name").reset_index(drop=True)
    result["total_spent"] = result["total_spent"].round(2)
    result["avg_per_txn"] = result["avg_per_txn"].round(2)
    return result


def category_mom(df: pd.DataFrame) -> dict:
    """Month-over-month spending comparison by category (last two months with data)."""
    expenses = df[
        df["is_expense"] & ~df["is_duplicate"] & ~df["category"].isin({"Transfer"})
    ]
    if expenses.empty:
        return {}
    months = sorted(expenses["year_month"].unique())
    if len(months) < 2:
        return {}
    curr_lbl, prev_lbl = months[-1], months[-2]
    curr = expenses[expenses["year_month"] == curr_lbl].groupby("category")["abs_amount"].sum()
    prev = expenses[expenses["year_month"] == prev_lbl].groupby("category")["abs_amount"].sum()
    all_cats = sorted(set(curr.index) | set(prev.index), key=lambda c: -curr.get(c, 0))
    return {
        "current_label": curr_lbl,
        "prev_label": prev_lbl,
        "categories": all_cats,
        "current": [round(float(curr.get(c, 0)), 2) for c in all_cats],
        "previous": [round(float(prev.get(c, 0)), 2) for c in all_cats],
    }


def budget_status(df: pd.DataFrame, budgets: dict[str, float]) -> list[dict]:
    """Compare current-month spending against budget limits.

    Returns a list of dicts sorted by % used descending.
    """
    if not budgets:
        return []
    expenses = df[
        df["is_expense"] & ~df["is_duplicate"] & ~df["category"].isin({"Transfer"})
    ]
    if expenses.empty:
        return []
    curr_month = expenses["year_month"].max()
    curr = expenses[expenses["year_month"] == curr_month].groupby("category")["abs_amount"].sum()
    rows = []
    for cat, limit in budgets.items():
        spent = float(curr.get(cat, 0))
        pct = round(spent / limit * 100, 1) if limit > 0 else 0
        rows.append({"category": cat, "budget": limit, "spent": round(spent, 2), "pct_used": pct})
    return sorted(rows, key=lambda r: -r["pct_used"])


def analyze(
    df: pd.DataFrame,
    starting_balance: float = 0.0,
    std_threshold: float = 2.0,
    budgets: dict[str, float] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Run the full analysis pipeline and return a dict of result DataFrames.

    Args:
        df: Transformed transaction DataFrame from transform module.
        starting_balance: Initial account balance for running balance calc.
        std_threshold: Z-score threshold for anomaly detection.

    Returns:
        Dict with keys:
            'transactions'        - full DataFrame with anomaly flags + running balance
            'monthly_by_category' - spending by year_month and category
            'monthly_summary'     - income vs expenses per month
            'anomalies'           - subset of flagged anomaly transactions
            'recurring'           - detected recurring transactions
    """
    print("Analyzing transactions...")

    df = flag_anomalies(df, std_threshold)
    df = running_balance(df, starting_balance)

    monthly_cat = monthly_spending_by_category(df)
    monthly_sum = monthly_income_vs_expenses(df)
    anomalies = df[df["is_anomaly"]].sort_values("abs_amount", ascending=False)
    recurring = find_recurring(df)

    print(f"  Months covered: {df['year_month'].nunique()}")
    print(f"  Anomalies detected: {len(anomalies)}")
    print(f"  Recurring transactions found: {len(recurring)}")

    if not monthly_sum.empty:
        total_income = monthly_sum["income"].sum()
        total_expenses = monthly_sum["expenses"].sum()
        print(f"  Total income:   ${total_income:,.2f}")
        print(f"  Total expenses: ${total_expenses:,.2f}")
        print(f"  Net:            ${total_income - total_expenses:,.2f}")

    merchants   = top_merchants(df)
    dow         = spending_by_dow(df)
    mom         = category_mom(df)
    bstatus     = budget_status(df, budgets or {})

    # Add savings rate to monthly summary
    if not monthly_sum.empty:
        monthly_sum = monthly_sum.copy()
        monthly_sum["savings_rate"] = (
            ((monthly_sum["income"] - monthly_sum["expenses"]) / monthly_sum["income"].replace(0, np.nan))
            .round(3)
            .fillna(0)
        )

    return {
        "transactions": df,
        "monthly_by_category": monthly_cat,
        "monthly_summary": monthly_sum,
        "anomalies": anomalies,
        "recurring": recurring,
        "top_merchants": merchants,
        "spending_by_dow": dow,
        "mom_comparison": mom,
        "budget_status": bstatus,
    }
