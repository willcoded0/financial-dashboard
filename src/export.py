"""
export.py - Export processed data to CSV and generate a self-contained HTML dashboard.
"""

import json
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# CSV exports
# ---------------------------------------------------------------------------

def export_csvs(results: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    exports = {
        "transactions_clean.csv":  results["transactions"],
        "monthly_summary.csv":     results["monthly_summary"],
        "monthly_by_category.csv": results["monthly_by_category"],
        "anomalies.csv":           results["anomalies"],
        "recurring.csv":           results["recurring"],
        "top_merchants.csv":       results["top_merchants"],
    }
    for filename, df in exports.items():
        path = output_dir / filename
        df.to_csv(path, index=False)
        print(f"  Saved {filename} ({len(df)} rows)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CAT_COLORS = {
    "Groceries":    "#38a169", "Fast Food":     "#e53e3e",
    "Dining":       "#dd6b20", "Gas":           "#d69e2e",
    "Subscriptions":"#805ad5", "Shopping":      "#3182ce",
    "Gaming":       "#00b5d8", "Entertainment": "#d53f8c",
    "Transportation":"#319795","Utilities":     "#718096",
    "Health":       "#48bb78", "Education":     "#4299e1",
    "Personal Care":"#ed64a6", "Insurance":     "#667eea",
    "Housing":      "#975a16", "Income":        "#276749",
    "Transfer":     "#4a5568", "GF Tax":        "#f687b3",
    "Brother":      "#f6ad55", "Mom / Car":     "#90cdf4",
    "Other":        "#2d3748",
}
_CHART_COLORS = [
    "#4C9BE8","#E8884C","#4CE8A0","#E84C6A","#9B4CE8",
    "#E8D44C","#4CE8D4","#E84CA0","#7BE84C","#E8B44C",
    "#B44CE8","#4CB4E8","#E8C44C","#4CE8B4","#E86C4C",
]

def _j(v) -> str:
    return json.dumps(v, default=str)

def _cc(cat: str) -> str:
    return _CAT_COLORS.get(cat, "#2d3748")


def _build_financial_context(results: dict) -> str:
    """Build a concise system prompt. Individual transactions are injected
    dynamically per-query in JavaScript to stay within the model's context window."""
    ms   = results["monthly_summary"]
    mc   = results["monthly_by_category"]

    total_income   = float(ms["income"].sum())   if not ms.empty else 0
    total_expenses = float(ms["expenses"].sum()) if not ms.empty else 0
    net            = total_income - total_expenses
    avg_rate       = float((ms.get("savings_rate", pd.Series([0.0])) * 100).mean()) if not ms.empty else 0
    months         = len(ms)

    # All category totals
    cat_totals = ""
    if not mc.empty:
        tc = mc.groupby("category")["total_spent"].sum().sort_values(ascending=False)
        cat_totals = "\n".join(f"  {cat}: ${amt:,.2f}" for cat, amt in tc.items())

    # Monthly breakdown (compact)
    monthly_lines = ""
    if not mc.empty:
        for ym in sorted(mc["year_month"].unique()):
            rows = mc[mc["year_month"] == ym].sort_values("total_spent", ascending=False)
            parts = " | ".join(f"{r['category']} ${r['total_spent']:,.2f}" for _, r in rows.iterrows())
            monthly_lines += f"  {ym}: {parts}\n"

    # Monthly income vs expenses
    monthly_summary = ""
    if not ms.empty:
        for _, r in ms.sort_values("year_month").iterrows():
            rate = float(r.get("savings_rate", 0) or 0) * 100
            monthly_summary += (
                f"  {r['year_month']}: income ${r['income']:,.2f}  "
                f"expenses ${r['expenses']:,.2f}  net ${r['net']:+,.2f}  ({rate:.0f}% saved)\n"
            )

    # Budget status
    budget_lines = ""
    for b in (results.get("budget_status") or []):
        flag = " !! OVER BUDGET" if b["spent"] > b["budget"] else f" ({b['pct_used']:.0f}% used)"
        budget_lines += f"  {b['category']}: ${b['spent']:.2f} / ${b['budget']:.2f}{flag}\n"

    # Recurring
    recurring_lines = ""
    if not results["recurring"].empty:
        total_rec = results["recurring"]["avg_amount"].sum()
        recurring_lines += f"  Est. monthly total: ${total_rec:.2f}\n"
        for _, r in results["recurring"].iterrows():
            freq = "weekly" if r["interval_days"] <= 8 else "bi-weekly" if r["interval_days"] <= 16 else "monthly"
            recurring_lines += f"  {r['merchant']}: ${r['avg_amount']:.2f}/{freq} ({r['category']})\n"

    return (
        "You are Puck, a sharp and direct personal financial assistant.\n"
        "You ONLY discuss personal finance. Answer from the data below — never make up numbers.\n"
        "When the user asks about specific transactions or a category/month, "
        "relevant transactions will be appended to their message automatically — use those to give exact answers.\n"
        "Be concise. No disclaimers. No generic advice unless asked.\n\n"
        f"OVERVIEW ({months} months of data):\n"
        f"  Total income:     ${total_income:,.2f}\n"
        f"  Total expenses:   ${total_expenses:,.2f}\n"
        f"  Net:              ${net:+,.2f}\n"
        f"  Avg savings rate: {avg_rate:.1f}%\n\n"
        f"ALL SPENDING CATEGORIES (all-time totals):\n{cat_totals}\n\n"
        f"MONTHLY SPENDING BY CATEGORY:\n{monthly_lines}\n"
        f"MONTHLY INCOME vs EXPENSES:\n{monthly_summary}\n"
        f"BUDGET STATUS (current month):\n{budget_lines}\n"
        f"RECURRING CHARGES:\n{recurring_lines}"
    )


def _current_month_spend(transactions: pd.DataFrame) -> dict:
    if transactions.empty:
        return {}
    curr = transactions["year_month"].max()
    exp = transactions[
        (transactions["year_month"] == curr)
        & transactions["is_expense"]
        & ~transactions["is_duplicate"]
        & ~transactions["category"].isin(["Transfer"])
    ]
    return exp.groupby("category")["abs_amount"].sum().round(2).to_dict()


# ---------------------------------------------------------------------------
# HTML dashboard
# ---------------------------------------------------------------------------

def generate_html_dashboard(results: dict, output_path: Path) -> None:
    txns         = results["transactions"]
    ms           = results["monthly_summary"]
    monthly_cat  = results["monthly_by_category"]
    anomalies    = results["anomalies"]
    recurring    = results["recurring"]
    merchants_df = results["top_merchants"]
    dow_df       = results["spending_by_dow"]
    mom          = results["mom_comparison"]
    bstatus      = results["budget_status"]

    # ── Summary stats ──────────────────────────────────────────────────────
    total_income   = float(ms["income"].sum())   if not ms.empty else 0
    total_expenses = float(ms["expenses"].sum()) if not ms.empty else 0
    net            = total_income - total_expenses
    avg_rate       = float((ms.get("savings_rate", pd.Series([0.0])) * 100).mean()) if not ms.empty else 0
    avg_monthly    = float(ms[ms["expenses"] > 0]["expenses"].mean()) if not ms.empty else 0
    biggest_cat    = ""
    if not txns.empty:
        exp = txns[txns["is_expense"] & ~txns["is_duplicate"] & ~txns["category"].isin(["Transfer"])]
        if not exp.empty:
            biggest_cat = exp.groupby("category")["abs_amount"].sum().idxmax()

    # ── Chart data ─────────────────────────────────────────────────────────
    trend_labels = ms["year_month"].tolist()
    income_data  = ms["income"].round(2).tolist()
    expense_data = ms["expenses"].round(2).tolist()

    pie_agg    = monthly_cat.groupby("category")["total_spent"].sum().sort_values(ascending=False)
    pie_labels = pie_agg.index.tolist()
    pie_data   = pie_agg.round(2).tolist()
    pie_colors = [_cc(c) for c in pie_labels]

    bal_df     = txns[["date","running_balance"]].sort_values("date").dropna()
    step       = max(1, len(bal_df) // 150)
    bal_s      = bal_df.iloc[::step]
    bal_labels = [str(d)[:10] for d in bal_s["date"]]
    bal_data   = bal_s["running_balance"].round(2).tolist()

    dow_labels = dow_df["day_name"].tolist()       if not dow_df.empty else []
    dow_totals = dow_df["total_spent"].tolist()    if not dow_df.empty else []
    dow_avgs   = dow_df["avg_per_txn"].tolist()    if not dow_df.empty else []

    mom_cats     = mom.get("categories", [])[:8]
    mom_curr     = mom.get("current",    [])[:8]
    mom_prev     = mom.get("previous",   [])[:8]
    mom_curr_lbl = mom.get("current_label", "")
    mom_prev_lbl = mom.get("prev_label",    "")

    # ── Budgets ────────────────────────────────────────────────────────────
    budgets_default = {b["category"]: b["budget"] for b in bstatus} if bstatus else {}
    curr_spend      = _current_month_spend(txns)

    # ── Top merchants table rows ───────────────────────────────────────────
    merch_rows = "".join(
        "<tr>"
        f"<td>{r['merchant']}</td>"
        f"<td><span class='badge' style='background:{_cc(r['category'])}'>{r['category']}</span></td>"
        f"<td class='num red'>-${r['total_spent']:,.2f}</td>"
        f"<td class='num dim'>{int(r['transactions'])}</td>"
        f"<td class='num dim'>${r['avg_amount']:.2f}</td>"
        "</tr>"
        for _, r in merchants_df.iterrows()
    )

    # ── Subscriptions list ─────────────────────────────────────────────────
    sub_total = float(recurring["avg_amount"].sum()) if not recurring.empty else 0
    sub_rows  = "".join(
        f"<div class='sub-row'>"
        f"<span class='sub-name'>{r['merchant']}</span>"
        f"<span class='sub-freq'>{'weekly' if r['interval_days']<=8 else 'bi-wk' if r['interval_days']<=16 else 'monthly'}</span>"
        f"<span class='red'>-${r['avg_amount']:.2f}</span>"
        f"</div>"
        for _, r in recurring.iterrows()
    ) if not recurring.empty else "<p class='dim'>None detected.</p>"

    # ── Anomaly cards ──────────────────────────────────────────────────────
    anom_cards = "".join(
        f"<div class='alert-card'>"
        f"<strong>{r.get('merchant',r['description'])}</strong>"
        f"<span class='red amt'>${r['abs_amount']:,.2f}</span>"
        f"<span class='meta'>{str(r['date'])[:10]} · {r.get('category','?')} · z={r.get('anomaly_zscore',0):.1f}σ</span>"
        f"</div>"
        for _, r in anomalies.head(9).iterrows()
    ) if not anomalies.empty else "<p class='dim'>None detected.</p>"

    # ── Transactions JSON ──────────────────────────────────────────────────
    txn_json = json.dumps([
        {"date": str(r["date"])[:10], "merchant": str(r.get("merchant", r["description"])),
         "category": str(r.get("category","Other")), "amount": float(r["amount"]),
         "dup": bool(r.get("is_duplicate",False)), "anomaly": bool(r.get("is_anomaly",False))}
        for _, r in txns.sort_values("date", ascending=False).iterrows()
    ], ensure_ascii=False)

    all_cats = sorted(txns["category"].dropna().unique().tolist())
    if "Transfer" in all_cats:
        all_cats.remove("Transfer"); all_cats.append("Transfer")

    date_range = ""
    if not txns.empty:
        dmin = txns["date"].min(); dmax = txns["date"].max()
        date_range = (f"{dmin.strftime('%b %Y') if hasattr(dmin,'strftime') else str(dmin)[:7]} – "
                      f"{dmax.strftime('%b %Y') if hasattr(dmax,'strftime') else str(dmax)[:7]}")

    financial_context = _build_financial_context(results)
    gen_time          = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    # ── HTML ───────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>Financial Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    :root{{
      --bg:#080808;--s1:#0e0e0e;--s2:#141414;
      --bd:#252525;--bdb:#3a3a3a;
      --txt:#e0dcd8;--txt2:#808080;--txt3:#4a4a4a;
      --blue:#c03030;--green:#38a858;--red:#d94040;
      --yellow:#c08830;--purple:#9040b0;--teal:#30a090;
      --r:13px;
    }}
    *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
    html{{font-size:14px;scrollbar-color:var(--bd) var(--bg);scrollbar-width:thin}}
    body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--txt);min-height:100vh}}

    /* ── Header ── */
    header{{
      background:linear-gradient(135deg,#080808 0%,#0d0d0d 60%,#090909 100%);
      padding:15px 26px;border-bottom:1px solid var(--bd);
      display:flex;gap:14px;align-items:center;
      position:relative;overflow:hidden;
    }}
    header::before{{
      content:'';position:absolute;inset:0;pointer-events:none;
      background:
        radial-gradient(ellipse 700px 200px at 8% 60%,rgba(192,48,48,.04),transparent),
        radial-gradient(ellipse 500px 160px at 92% 20%,rgba(140,30,30,.03),transparent);
    }}
    .hdr-ascii{{
      font-family:'Courier New',monospace;font-size:6.5px;line-height:1.1;
      color:#6b1212;white-space:pre;flex-shrink:0;align-self:center;
      filter:drop-shadow(0 0 8px rgba(160,20,20,.35));
    }}
    .hdr-text h1{{font-size:1.28rem;font-weight:800;color:#fff;letter-spacing:-.02em;position:relative}}
    .hdr-text p{{font-size:.75rem;color:var(--txt2);margin-top:2px;position:relative}}
    .hdr-badge{{
      margin-left:auto;display:inline-flex;align-items:center;gap:6px;
      background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.1);
      border-radius:99px;padding:4px 12px;font-size:.7rem;color:var(--txt2);
      position:relative;white-space:nowrap;
    }}

    /* ── Layout ── */
    .page{{padding:18px 22px;max-width:1860px;margin:0 auto}}

    /* ── Summary Cards ── */
    .cards{{display:flex;gap:12px;margin-bottom:18px;flex-wrap:wrap}}
    .card{{
      flex:1;min-width:130px;
      background:var(--s1);border:1px solid var(--bd);
      border-radius:var(--r);padding:14px 17px;
      position:relative;overflow:hidden;
      transition:transform .2s ease,box-shadow .2s ease,border-color .2s;
      cursor:default;
    }}
    .card::before{{
      content:'';position:absolute;top:0;left:0;right:0;
      height:3px;border-radius:var(--r) var(--r) 0 0;
    }}
    .card::after{{
      content:'';position:absolute;top:-36px;right:-18px;
      width:78px;height:78px;border-radius:50%;opacity:.07;
    }}
    .card:hover{{transform:translateY(-3px);box-shadow:0 10px 30px rgba(0,0,0,.55)}}
    .card-label{{font-size:.64rem;font-weight:600;color:var(--txt3);text-transform:uppercase;letter-spacing:.07em}}
    .card-icon{{font-size:1.1rem;position:absolute;top:11px;right:13px;opacity:.35}}
    .card-value{{font-size:1.42rem;font-weight:800;margin-top:7px;letter-spacing:-.02em;line-height:1;white-space:nowrap}}
    .card-income::before{{background:linear-gradient(90deg,var(--green),transparent)}}
    .card-income::after{{background:var(--green)}}
    .card-income:hover{{border-color:rgba(15,196,114,.3)}}
    .card-expenses::before{{background:linear-gradient(90deg,var(--red),transparent)}}
    .card-expenses::after{{background:var(--red)}}
    .card-expenses:hover{{border-color:rgba(240,64,96,.3)}}
    .card-net-pos::before{{background:linear-gradient(90deg,var(--blue),transparent)}}
    .card-net-pos::after{{background:var(--blue)}}
    .card-net-pos:hover{{border-color:rgba(192,48,48,.3)}}
    .card-net-neg::before{{background:linear-gradient(90deg,var(--red),transparent)}}
    .card-net-neg::after{{background:var(--red)}}
    .card-savings::before{{background:linear-gradient(90deg,var(--teal),transparent)}}
    .card-savings::after{{background:var(--teal)}}
    .card-savings:hover{{border-color:rgba(20,184,192,.3)}}
    .card-spend::before{{background:linear-gradient(90deg,var(--yellow),transparent)}}
    .card-spend::after{{background:var(--yellow)}}
    .card-spend:hover{{border-color:rgba(245,160,32,.3)}}
    .card-cat::before{{background:linear-gradient(90deg,var(--purple),transparent)}}
    .card-cat::after{{background:var(--purple)}}
    .card-cat:hover{{border-color:rgba(138,88,240,.3)}}
    .card-months::before{{background:linear-gradient(90deg,#7ab0e8,transparent)}}
    .card-months::after{{background:#7ab0e8}}
    .card-months:hover{{border-color:rgba(122,176,232,.3)}}

    /* ── Panel ── */
    .main{{display:grid;grid-template-columns:1fr 330px;gap:16px;align-items:start;margin-bottom:16px}}
    .charts-area{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
    .panel{{background:var(--s1);border:1px solid var(--bd);border-radius:var(--r);padding:18px;transition:box-shadow .2s}}
    .panel:hover{{box-shadow:0 6px 32px rgba(0,0,0,.5)}}
    .panel h2{{
      font-size:.68rem;font-weight:700;color:var(--txt2);
      text-transform:uppercase;letter-spacing:.08em;
      margin-bottom:14px;display:flex;align-items:center;gap:7px;
    }}
    .panel h2::before{{content:'';display:inline-block;width:3px;height:12px;border-radius:2px;background:var(--blue);flex-shrink:0}}
    .span2{{grid-column:1/span 2}}
    .ch{{position:relative;height:220px}}
    .ch-wide{{position:relative;height:220px}}
    /* ── Chart Carousel ── */
    .car-nav{{display:flex;align-items:center;gap:10px;margin-bottom:14px}}
    .car-btn{{
      background:var(--s2);border:1px solid var(--bd);border-radius:9px;
      color:var(--txt2);width:34px;height:34px;cursor:pointer;flex-shrink:0;
      font-size:1.1rem;display:flex;align-items:center;justify-content:center;
      transition:.15s;user-select:none;
    }}
    .car-btn:hover{{background:var(--bdb);color:var(--txt);border-color:var(--bdb)}}
    .car-info{{flex:1;text-align:center}}
    .car-title{{font-size:.82rem;font-weight:600;color:var(--txt);display:block}}
    .car-dots{{display:flex;gap:5px;justify-content:center;margin-top:6px}}
    .car-dot{{width:6px;height:6px;border-radius:50%;background:var(--bd);cursor:pointer;transition:width .2s,background .2s}}
    .car-dot.on{{background:var(--blue);width:20px;border-radius:3px}}
    .car-slide{{display:none}}
    .car-slide.on{{display:block}}

    /* ── Budget ── */
    .budget-row{{margin-bottom:13px}}
    .budget-meta{{display:flex;align-items:center;gap:8px;margin-bottom:6px;flex-wrap:wrap}}
    .budget-spent{{font-size:.79rem;color:var(--txt)}}
    .budget-of{{font-size:.72rem;color:var(--txt3)}}
    .budget-input{{
      background:transparent;border:none;border-bottom:1px dashed var(--bd);border-radius:0;
      color:var(--txt);font-size:.82rem;width:64px;padding:1px 2px;
      text-align:right;transition:border-color .15s;
    }}
    .budget-input:focus{{outline:none;border-bottom-color:var(--blue)}}
    .budget-input::-webkit-inner-spin-button,.budget-input::-webkit-outer-spin-button{{opacity:.3}}
    .budget-pct{{font-size:.7rem;color:var(--txt3);margin-left:auto;font-variant-numeric:tabular-nums}}
    .budget-track{{background:#0a0a0a;border-radius:9999px;height:8px;overflow:hidden;box-shadow:inset 0 1px 3px rgba(0,0,0,.5)}}
    .budget-fill{{height:100%;border-radius:9999px;transition:width .65s cubic-bezier(.4,0,.2,1)}}
    .save-btn{{padding:5px 13px;background:var(--s2);border:1px solid var(--bd);border-radius:7px;color:var(--txt2);font-size:.73rem;cursor:pointer;transition:.15s;margin-left:5px}}
    .save-btn:hover{{background:var(--bdb);color:var(--txt)}}
    .saved-note{{font-size:.69rem;color:var(--green);margin-left:5px;opacity:0;transition:opacity .3s}}

    /* ── Subscriptions ── */
    .sub-total{{font-size:.8rem;color:var(--txt2);padding-bottom:9px;margin-bottom:9px;border-bottom:1px solid var(--bd)}}
    .sub-row{{display:flex;align-items:center;gap:8px;padding:7px 0;border-bottom:1px solid rgba(37,37,37,.6);font-size:.83rem}}
    .sub-row:last-child{{border-bottom:none}}
    .sub-name{{flex:1;color:var(--txt)}}
    .sub-freq{{font-size:.66rem;color:var(--txt3);background:var(--s2);padding:2px 7px;border-radius:9999px;white-space:nowrap;border:1px solid var(--bd)}}

    /* ── Chat Panel ── */
    .chat-panel{{
      background:var(--s1);border:1px solid var(--bd);border-radius:var(--r);
      display:flex;flex-direction:column;
      position:sticky;top:16px;
      height:calc(100vh - 84px);min-height:520px;max-height:920px;
    }}
    .chat-header{{
      padding:15px 16px;border-bottom:1px solid var(--bd);flex-shrink:0;
      background:linear-gradient(135deg,#0c0c0c,#111111);
      border-radius:var(--r) var(--r) 0 0;
    }}
    .chat-header-top{{display:flex;align-items:center;gap:8px}}
    .chat-header h2{{font-size:.82rem;font-weight:700;color:var(--txt);letter-spacing:.01em;margin-bottom:0}}
    .chat-header h2::before{{display:none}}
    .chat-status{{font-size:.67rem;color:var(--txt3);margin-top:5px;display:flex;align-items:center;gap:5px}}
    .sdot{{width:6px;height:6px;border-radius:50%;background:var(--txt3);display:inline-block;transition:background .4s,box-shadow .4s;flex-shrink:0}}
    .sdot.online{{background:var(--green);box-shadow:0 0 7px var(--green);animation:pulse-dot 2.5s infinite}}
    @keyframes pulse-dot{{0%,100%{{opacity:1}}50%{{opacity:.35}}}}
    .chat-messages{{flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:10px}}
    .chat-messages::-webkit-scrollbar{{width:3px}}
    .chat-messages::-webkit-scrollbar-track{{background:transparent}}
    .chat-messages::-webkit-scrollbar-thumb{{background:var(--bd);border-radius:2px}}
    .msg{{max-width:93%;padding:10px 13px;border-radius:12px;font-size:.83rem;line-height:1.55;word-wrap:break-word;white-space:pre-wrap;animation:msg-in .22s ease}}
    @keyframes msg-in{{from{{opacity:0;transform:translateY(5px)}}to{{opacity:1;transform:none}}}}
    .msg.user{{background:linear-gradient(135deg,#1c1c1c,#232323);border:1px solid #383838;align-self:flex-end;color:var(--txt)}}
    .msg.assistant{{background:var(--s2);border:1px solid var(--bd);align-self:flex-start;color:var(--txt)}}
    .msg.assistant.streaming::after{{content:"▋";animation:blink .8s infinite}}
    @keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:0}}}}
    .chat-input-area{{padding:11px;border-top:1px solid var(--bd);flex-shrink:0;display:flex;gap:8px}}
    .chat-input-area textarea{{
      flex:1;background:var(--bg);border:1px solid var(--bd);border-radius:9px;
      color:var(--txt);font-size:.83rem;padding:8px 10px;resize:none;height:56px;
      outline:none;font-family:inherit;line-height:1.4;transition:border-color .15s,box-shadow .15s;
    }}
    .chat-input-area textarea:focus{{border-color:var(--blue);box-shadow:0 0 0 2px rgba(192,48,48,.15)}}
    .send-btn{{
      background:linear-gradient(135deg,#1e1e1e,#181818);
      border:1px solid #383838;border-radius:9px;color:var(--txt2);
      font-size:1rem;padding:0 14px;cursor:pointer;transition:.15s;flex-shrink:0;
    }}
    .send-btn:hover{{background:linear-gradient(135deg,#2a2a2a,#222222);color:var(--txt)}}
    .send-btn:disabled{{opacity:.35;cursor:not-allowed}}
    .chat-hint{{font-size:.63rem;color:var(--txt3);padding:0 12px 8px;flex-shrink:0;text-align:center}}

    /* ── Lower grid ── */
    .lower{{display:grid;grid-template-columns:3fr 2fr;gap:14px;margin-bottom:16px}}
    .anom-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px}}
    .alert-card{{
      background:linear-gradient(135deg,#0f0f0f,#141414);
      border:1px solid #2a2a2a;border-radius:11px;padding:12px 14px;
      transition:border-color .2s,box-shadow .2s;
    }}
    .alert-card:hover{{border-color:rgba(192,48,48,.4);box-shadow:0 0 22px rgba(192,48,48,.08)}}
    .alert-card strong{{display:block;font-size:.83rem;color:var(--txt);margin-bottom:4px}}
    .amt{{font-size:1.05rem;font-weight:700;display:block;color:var(--red)}}
    .meta{{font-size:.67rem;opacity:.6;display:block;margin-top:3px}}

    /* ── Transactions ── */
    .txn-controls{{display:flex;flex-wrap:wrap;gap:9px;align-items:center;margin-bottom:10px}}
    .search-wrap{{position:relative;flex:1;min-width:180px}}
    .search-wrap input{{
      width:100%;background:var(--bg);border:1px solid var(--bd);border-radius:8px;
      padding:7px 10px 7px 30px;color:var(--txt);font-size:.83rem;outline:none;
      transition:border-color .15s,box-shadow .15s;
    }}
    .search-wrap input:focus{{border-color:var(--blue);box-shadow:0 0 0 2px rgba(192,48,48,.15)}}
    .si{{position:absolute;left:9px;top:50%;transform:translateY(-50%);color:var(--txt3);font-size:.8rem;pointer-events:none}}
    .txn-count{{font-size:.72rem;color:var(--txt3);white-space:nowrap}}
    .filter-row{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}}
    .fb{{
      padding:3px 10px;border-radius:9999px;border:1px solid var(--bd);
      background:var(--s1);color:var(--txt2);font-size:.7rem;cursor:pointer;
      transition:.15s;white-space:nowrap;
    }}
    .fb:hover{{border-color:var(--bdb);color:var(--txt)}}
    .fb.on{{
      border-color:var(--cc,var(--blue));
      background:color-mix(in srgb,var(--cc,var(--blue)) 15%,transparent);
      color:#fff;box-shadow:0 0 10px color-mix(in srgb,var(--cc,var(--blue)) 25%,transparent);
    }}
    .txn-scroll{{max-height:460px;overflow-y:auto;border-radius:9px;border:1px solid var(--bd);background:var(--bg)}}
    .txn-scroll::-webkit-scrollbar{{width:3px}}
    .txn-scroll::-webkit-scrollbar-track{{background:var(--bg)}}
    .txn-scroll::-webkit-scrollbar-thumb{{background:var(--bd);border-radius:2px}}
    table{{width:100%;border-collapse:collapse;font-size:.83rem}}
    thead th{{
      position:sticky;top:0;background:#0e0e0e;text-align:left;
      padding:9px 12px;border-bottom:1px solid var(--bd);
      color:var(--txt3);font-size:.66rem;text-transform:uppercase;letter-spacing:.05em;z-index:1;
    }}
    th.sort{{cursor:pointer;user-select:none;transition:color .15s}}
    th.sort:hover{{color:var(--txt)}}
    td{{padding:8px 12px;border-bottom:1px solid rgba(37,37,37,.4);vertical-align:middle}}
    tbody tr:last-child td{{border-bottom:none}}
    tbody tr:hover td{{background:rgba(255,255,255,.03)}}
    .badge{{
      display:inline-block;padding:2px 8px;border-radius:9999px;
      font-size:.67rem;font-weight:600;color:#fff;white-space:nowrap;
      text-shadow:0 1px 2px rgba(0,0,0,.45);
    }}
    .flag{{display:inline-block;padding:1px 4px;border-radius:3px;font-size:.57rem;font-weight:700;vertical-align:middle;margin-left:3px}}
    .dup{{background:#5a3800;color:#fbd38d}}
    .anom{{background:#5a1f1f;color:#fc8181}}
    .no-rows{{padding:28px;text-align:center;color:var(--txt3);font-style:italic;font-size:.83rem}}

    /* ── Utilities ── */
    .green{{color:var(--green)}} .red{{color:var(--red)}} .dim{{color:var(--txt3);font-style:italic;font-size:.8rem}}
    .num{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
    footer{{text-align:center;padding:16px;color:var(--txt3);font-size:.67rem;border-top:1px solid var(--bd);margin-top:8px}}
    code{{background:var(--s2);padding:1px 5px;border-radius:4px;font-size:.78em}}
    .merch-scroll{{max-height:360px;overflow-y:auto}}
    .merch-scroll::-webkit-scrollbar{{width:3px}}
    .merch-scroll::-webkit-scrollbar-track{{background:var(--bg)}}
    .merch-scroll::-webkit-scrollbar-thumb{{background:var(--bd);border-radius:2px}}
  </style>
</head>
<body>
<header>
  <pre class="hdr-ascii">⠀⠀  ⢀⡴⠁⠀⠀⣿⡏⠀⠀⠱⣄
⠀⠀⢀⣴⡟⠁⠀⠀⠀⣿⡇⠀⠀⠀⠙⣷⣄
⠀⠀⠙⢿⣷⣄⠀⠀⠀⣿⡇⠀⠀⢀⣴⣿⠋
⠀⠀⠀⠀⠙⢿⣷⣄⠀⢻⡇⢀⣴⣿⠋
⠀⠀⠀⠀⠀⠀⠈⠻⣷⣾⣷⡿⠋
⠀⠀⠀⠀⠀⠀⠀⢀⣼⣿⣿⣷⣄
⠀⠀⠀⠀⠀⢀⣶⣿⠟⢹⣏⠻⢿⣷⣄
⠀⠀⠀⢀⣼⣿⠟⠁⠀⢸⣿⠀⠈⠙⢿⣷⣄
⠀⠀⣴⣿⡟⠁⠀⠀⠀⢸⣿⠀⠀⠀⠀⣹⣿⡷
⠀⠀⠈⠻⣿⣦⡀⠀⠀⢸⣿⠀⠀⢀⣼⣿⠏
⠀⠀⠀⠀⠈⠻⣿⣦⡀⢸⣿⠀⣴⣿⠟⠁
⠀⠀⠀⠀⠀⠀⠈⠻⣿⣾⣿⣾⡿⠃⠀
⠀⠀ ⠀⠀⠀⠀⠀⠈⠻⡿⠋</pre>
  <div class="hdr-text">
    <h1>Financial Dashboard</h1>
    <p>Personal spending &amp; income analysis</p>
  </div>
  <div class="hdr-badge">{date_range} &nbsp;&bull;&nbsp; {len(txns)} transactions</div>
</header>

<div class="page">

<!-- ── Summary cards ─────────────────────────────────────────────────── -->
<div class="cards">
  <div class="card card-income">
    <div class="card-label">Total Income</div>
    <div class="card-value green" data-val="{total_income:.0f}" data-prefix="$">${total_income:,.0f}</div>
  </div>
  <div class="card card-expenses">
    <div class="card-label">Total Expenses</div>
    <div class="card-value red" data-val="{total_expenses:.0f}" data-prefix="$">${total_expenses:,.0f}</div>
  </div>
  <div class="card {'card-net-pos' if net>=0 else 'card-net-neg'}">
    <div class="card-label">Net Savings</div>
    <div class="card-value {'green' if net>=0 else 'red'}" data-val="{abs(net):.0f}" data-prefix="{'$' if net>=0 else '-$'}">${net:+,.0f}</div>
  </div>
  <div class="card card-savings">
    <div class="card-label">Avg Savings Rate</div>
    <div class="card-value {'green' if avg_rate>=0 else 'red'}" data-val="{avg_rate:.1f}" data-suffix="%">{avg_rate:.1f}%</div>
  </div>
  <div class="card card-spend">
    <div class="card-label">Avg Monthly Spend</div>
    <div class="card-value" data-val="{avg_monthly:.0f}" data-prefix="$">${avg_monthly:,.0f}</div>
  </div>
  <div class="card card-cat">
    <div class="card-label">Top Category</div>
    <div class="card-value" style="font-size:.95rem;margin-top:9px">{biggest_cat}</div>
  </div>
  <div class="card card-months">
    <div class="card-label">Months of Data</div>
    <div class="card-value" data-val="{len(ms)}">{len(ms)}</div>
  </div>
</div>

<!-- ── Main grid (charts + chat) ─────────────────────────────────────── -->
<div class="main">

  <!-- Charts area (2-col inner grid) -->
  <div class="charts-area">

    <!-- Chart carousel -->
    <div class="panel span2">
      <div class="car-nav">
        <button class="car-btn" id="carPrev">&#8592;</button>
        <div class="car-info">
          <span class="car-title" id="carTitle"></span>
          <div class="car-dots" id="carDots"></div>
        </div>
        <button class="car-btn" id="carNext">&#8594;</button>
      </div>
      <div class="car-slide on" id="car-0"><div class="ch"><canvas id="trendChart"></canvas></div></div>
      <div class="car-slide"    id="car-1"><div class="ch"><canvas id="pieChart"></canvas></div></div>
      <div class="car-slide"    id="car-2"><div class="ch"><canvas id="momChart"></canvas></div></div>
      <div class="car-slide"    id="car-3"><div class="ch"><canvas id="dowChart"></canvas></div></div>
      <div class="car-slide"    id="car-4"><div class="ch-wide"><canvas id="balChart"></canvas></div></div>
    </div>

    <div class="panel">
      <h2>Budget Tracker — Current Month
        <button class="save-btn" onclick="saveBudgets()" style="margin-left:8px">Save</button>
        <span class="saved-note" id="savedNote">Saved ✓</span>
      </h2>
      <div id="budgetContainer"></div>
    </div>

    <div class="panel">
      <h2>Subscriptions &amp; Recurring</h2>
      <div class="sub-total">Est. monthly: <strong class="red">-${sub_total:.2f}</strong></div>
      {sub_rows}
    </div>

  </div><!-- end charts-area -->

  <!-- Sticky chat panel -->
  <div class="chat-panel">
    <div class="chat-header">
      <div class="chat-header-top">
        <h2>Puck — Financial Assistant</h2>
      </div>
      <div class="chat-status"><span class="sdot" id="statusDot"></span><span id="chatStatus">Connecting to Ollama…</span></div>
    </div>
    <div class="chat-messages" id="chatMessages">
      <div class="msg assistant">Hey! I'm Puck, your financial assistant. I have your spending data loaded — ask me anything about your finances.</div>
    </div>
    <div class="chat-input-area">
      <textarea id="chatInput" placeholder="Ask about your spending, savings, budgets…" onkeydown="handleKey(event)"></textarea>
      <button class="send-btn" id="sendBtn" onclick="sendMessage()" title="Send (Enter)">&#9658;</button>
    </div>
    <div class="chat-hint">Enter to send &bull; Shift+Enter for newline</div>
  </div>

</div><!-- end main grid -->

<!-- ── Lower: merchants + anomalies ──────────────────────────────────── -->
<div class="lower">
  <div class="panel">
    <h2>Top Merchants by Spend</h2>
    <div class="merch-scroll">
      <table>
        <thead><tr>
          <th>Merchant</th><th>Category</th>
          <th class="num">Total</th>
          <th class="num">Txns</th>
          <th class="num">Avg</th>
        </tr></thead>
        <tbody>{merch_rows}</tbody>
      </table>
    </div>
  </div>
  <div class="panel">
    <h2>Anomaly Alerts</h2>
    <div class="anom-grid">{anom_cards}</div>
  </div>
</div>

<!-- ── Transactions ───────────────────────────────────────────────────── -->
<div class="panel">
  <h2>All Transactions</h2>
  <div class="txn-controls">
    <div class="search-wrap">
      <span class="si">&#128269;</span>
      <input id="txnSearch" type="text" placeholder="Search merchant…" oninput="applyFilters()"/>
    </div>
    <span class="txn-count" id="txnCount"></span>
  </div>
  <div class="filter-row" id="filterRow"></div>
  <div class="txn-scroll">
    <table>
      <thead><tr>
        <th class="sort" onclick="sortBy('date')">Date &#8597;</th>
        <th>Merchant</th><th>Category</th>
        <th class="sort num" onclick="sortBy('amount')">Amount &#8597;</th>
      </tr></thead>
      <tbody id="txnBody"></tbody>
    </table>
    <div class="no-rows" id="noRows" style="display:none">No transactions match.</div>
  </div>
</div>

</div><!-- end page -->
<footer>Generated {gen_time}</footer>

<script>
// ── Data ─────────────────────────────────────────────────────────────────
const CAT_COLORS    = {_j(_CAT_COLORS)};
const CHART_COLORS  = {_j(_CHART_COLORS)};
const TRANSACTIONS  = {txn_json};
const BUDGETS_DEF   = {_j(budgets_default)};
const CURR_SPEND    = {_j(curr_spend)};
const SYSTEM_PROMPT = {_j(financial_context)};

function cc(cat) {{ return CAT_COLORS[cat] || "#374d68"; }}
Chart.defaults.color = "#808080";
Chart.defaults.borderColor = "#252525";
Chart.defaults.font.family = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif";

// ── Animated counters ────────────────────────────────────────────────────
(function animateCards() {{
  document.querySelectorAll(".card-value[data-val]").forEach(el => {{
    const end    = parseFloat(el.dataset.val);
    const prefix = el.dataset.prefix || "";
    const suffix = el.dataset.suffix || "";
    const dur    = 900;
    const start  = performance.now();
    const isInt  = !String(end).includes(".");
    function tick(now) {{
      const p = Math.min((now - start) / dur, 1);
      const ease = 1 - Math.pow(1 - p, 3);
      const v = end * ease;
      const fmt = isInt ? Math.round(v).toLocaleString() : v.toFixed(1);
      el.textContent = prefix + fmt + suffix;
      if (p < 1) requestAnimationFrame(tick);
    }}
    requestAnimationFrame(tick);
  }});
}})();

// ── Charts ────────────────────────────────────────────────────────────────
new Chart(document.getElementById("trendChart"), {{
  type:"line",
  data:{{labels:{_j(trend_labels)},datasets:[
    {{label:"Income",  data:{_j(income_data)},  borderColor:"#0fc472",backgroundColor:"rgba(15,196,114,.07)",borderWidth:2,tension:.35,fill:true,pointRadius:3,pointBackgroundColor:"#0fc472"}},
    {{label:"Expenses",data:{_j(expense_data)}, borderColor:"#f04060",backgroundColor:"rgba(240,64,96,.07)",borderWidth:2,tension:.35,fill:true,pointRadius:3,pointBackgroundColor:"#f04060"}},
  ]}},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{labels:{{color:"#e0dcd8",font:{{size:11}}}}}}}},
    scales:{{
      x:{{ticks:{{color:"#808080",maxRotation:45,font:{{size:10}}}},grid:{{color:"rgba(37,37,37,.5)"}}}},
      y:{{ticks:{{color:"#808080",callback:v=>"$"+v.toLocaleString(),font:{{size:10}}}},grid:{{color:"rgba(37,37,37,.5)"}}}}
    }}
  }}
}});

new Chart(document.getElementById("pieChart"), {{
  type:"doughnut",
  data:{{labels:{_j(pie_labels)},datasets:[{{data:{_j(pie_data)},backgroundColor:{_j(pie_colors)},borderColor:"#060911",borderWidth:2}}]}},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{
      legend:{{position:"right",labels:{{color:"#e0dcd8",padding:9,font:{{size:10}}}}}},
      tooltip:{{callbacks:{{label:c=>` ${{c.label}}: $${{c.parsed.toLocaleString(undefined,{{minimumFractionDigits:2}})}}`}}}}
    }}
  }}
}});

new Chart(document.getElementById("momChart"), {{
  type:"bar", indexAxis:"y",
  data:{{labels:{_j(mom_cats)},datasets:[
    {{label:"{mom_curr_lbl}",data:{_j(mom_curr)},backgroundColor:"rgba(192,48,48,.85)",borderRadius:3}},
    {{label:"{mom_prev_lbl}",data:{_j(mom_prev)},backgroundColor:"rgba(255,255,255,.1)",borderRadius:3}},
  ]}},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{labels:{{color:"#e0dcd8",font:{{size:10}}}}}}}},
    scales:{{
      x:{{ticks:{{color:"#808080",callback:v=>"$"+v,font:{{size:10}}}},grid:{{color:"rgba(37,37,37,.5)"}}}},
      y:{{ticks:{{color:"#e0dcd8",font:{{size:10}}}},grid:{{display:false}}}}
    }}
  }}
}});

new Chart(document.getElementById("dowChart"), {{
  type:"bar",
  data:{{labels:{_j(dow_labels)},datasets:[
    {{label:"Total",data:{_j(dow_totals)},backgroundColor:"rgba(192,48,48,.7)",borderRadius:4,yAxisID:"y"}},
    {{label:"Avg/txn",data:{_j(dow_avgs)},type:"line",borderColor:"#f5a020",backgroundColor:"transparent",borderWidth:2,pointRadius:4,pointBackgroundColor:"#f5a020",tension:.3,yAxisID:"y2"}},
  ]}},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{labels:{{color:"#e0dcd8",font:{{size:10}}}}}}}},
    scales:{{
      x:{{ticks:{{color:"#808080",font:{{size:10}}}},grid:{{color:"rgba(37,37,37,.5)"}}}},
      y:{{ticks:{{color:"#808080",callback:v=>"$"+v,font:{{size:10}}}},grid:{{color:"rgba(37,37,37,.5)"}}}},
      y2:{{position:"right",ticks:{{color:"#808080",callback:v=>"$"+v,font:{{size:10}}}},grid:{{display:false}}}}
    }}
  }}
}});

new Chart(document.getElementById("balChart"), {{
  type:"line",
  data:{{labels:{_j(bal_labels)},datasets:[{{
    label:"Balance",data:{_j(bal_data)},
    borderColor:"#8a58f0",backgroundColor:"rgba(138,88,240,.08)",
    borderWidth:2,tension:.3,fill:true,pointRadius:0
  }}]}},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{display:false}}}},
    scales:{{
      x:{{ticks:{{color:"#808080",maxTicksLimit:12,font:{{size:10}}}},grid:{{color:"rgba(37,37,37,.5)"}}}},
      y:{{ticks:{{color:"#808080",callback:v=>"$"+v.toLocaleString(),font:{{size:10}}}},grid:{{color:"rgba(37,37,37,.5)"}}}}
    }}
  }}
}});

// ── Chart Carousel ────────────────────────────────────────────────────────
const CAR_SLIDES = [
  {{id:"car-0", chartId:"trendChart", title:"Monthly Income vs Expenses"}},
  {{id:"car-1", chartId:"pieChart",   title:"Spending by Category"}},
  {{id:"car-2", chartId:"momChart",   title:"Month-over-Month: {mom_curr_lbl} vs {mom_prev_lbl}"}},
  {{id:"car-3", chartId:"dowChart",   title:"Spending by Day of Week"}},
  {{id:"car-4", chartId:"balChart",   title:"Running Balance"}},
];
let carIdx = 0;

// Build dots
const dotsEl = document.getElementById("carDots");
CAR_SLIDES.forEach((_, i) => {{
  const d = document.createElement("div");
  d.className = "car-dot" + (i === 0 ? " on" : "");
  d.onclick = () => showSlide(i);
  dotsEl.appendChild(d);
}});

function showSlide(idx) {{
  carIdx = ((idx % CAR_SLIDES.length) + CAR_SLIDES.length) % CAR_SLIDES.length;
  CAR_SLIDES.forEach((s, i) => {{
    document.getElementById(s.id).classList.toggle("on", i === carIdx);
  }});
  document.querySelectorAll(".car-dot").forEach((d, i) => d.classList.toggle("on", i === carIdx));
  document.getElementById("carTitle").textContent = CAR_SLIDES[carIdx].title;
  // Resize so Chart.js fills the newly-visible canvas
  setTimeout(() => {{
    const chart = Chart.getChart(CAR_SLIDES[carIdx].chartId);
    if (chart) chart.resize();
  }}, 0);
}}

document.getElementById("carPrev").onclick = () => showSlide(carIdx - 1);
document.getElementById("carNext").onclick = () => showSlide(carIdx + 1);
document.addEventListener("keydown", e => {{
  if (document.activeElement.tagName === "INPUT" || document.activeElement.tagName === "TEXTAREA") return;
  if (e.key === "ArrowLeft")  showSlide(carIdx - 1);
  if (e.key === "ArrowRight") showSlide(carIdx + 1);
}});
showSlide(0);

// ── Budget editor ─────────────────────────────────────────────────────────
let budgets = {{}};
function loadBudgets() {{
  budgets = {{...BUDGETS_DEF}};
  const stored = localStorage.getItem("fd_budgets_v2");
  if (stored) Object.assign(budgets, JSON.parse(stored));
  renderBudgets();
}}
function saveBudgets() {{
  localStorage.setItem("fd_budgets_v2", JSON.stringify(budgets));
  const note = document.getElementById("savedNote");
  note.style.opacity = "1";
  setTimeout(() => note.style.opacity = "0", 1800);
  renderBudgets();
}}
function updateBudget(cat, val) {{
  budgets[cat] = Math.max(0, parseFloat(val) || 0);
}}
function renderBudgets() {{
  const c = document.getElementById("budgetContainer");
  if (!Object.keys(budgets).length) {{
    c.innerHTML = "<p class='dim'>No budgets configured in categories.yaml.</p>"; return;
  }}
  const entries = Object.entries(budgets).sort((a,b) => {{
    const pa = (CURR_SPEND[a[0]]||0)/a[1]*100;
    const pb = (CURR_SPEND[b[0]]||0)/b[1]*100;
    return pb - pa;
  }});
  c.innerHTML = entries.map(([cat, limit]) => {{
    const spent = CURR_SPEND[cat] || 0;
    const pct   = Math.min(spent / limit * 100, 100);
    const over  = spent > limit;
    const clr = over
      ? "linear-gradient(90deg,#f04060,#f06080)"
      : pct > 80
        ? "linear-gradient(90deg,#f5a020,#f5c040)"
        : `linear-gradient(90deg,${{cc(cat)}},color-mix(in srgb,${{cc(cat)}} 60%,#14b8c0))`;
    return `
    <div class="budget-row">
      <div class="budget-meta">
        <span class="badge" style="background:${{cc(cat)}}">${{cat}}</span>
        <span class="budget-spent ${{over?'red':''}}">${{over?"⚠ ":""}}$${{spent.toFixed(0)}}</span>
        <span class="budget-of">of $</span>
        <input type="number" class="budget-input" value="${{limit}}" min="0" step="10"
               onchange="updateBudget('${{cat}}',this.value)" title="Edit budget"/>
        <span class="budget-pct">${{pct.toFixed(0)}}%</span>
      </div>
      <div class="budget-track"><div class="budget-fill" style="width:${{pct}}%;background:${{clr}}"></div></div>
    </div>`;
  }}).join("");
}}
loadBudgets();

// ── Puck Chat ─────────────────────────────────────────────────────────────
let chatHistory = [];
let isStreaming = false;
let msgCounter  = 0;

async function checkOllama() {{
  try {{
    const r = await fetch("http://localhost:11434/api/tags", {{signal: AbortSignal.timeout(3000)}});
    if (r.ok) {{
      document.getElementById("chatStatus").textContent = "puck · ready";
      document.getElementById("statusDot").classList.add("online");
    }}
  }} catch {{
    document.getElementById("chatStatus").textContent = "⚠ offline — serve via HTTP (see readme)";
    document.getElementById("statusDot").style.background = "var(--red)";
  }}
}}
checkOllama();

function appendMsg(role, text) {{
  const id = "msg-" + (++msgCounter);
  const div = document.createElement("div");
  div.className = "msg " + role;
  div.id = id;
  div.textContent = text;
  document.getElementById("chatMessages").appendChild(div);
  div.scrollIntoView({{behavior:"smooth", block:"end"}});
  return id;
}}
function updateMsg(id, text) {{
  const el = document.getElementById(id);
  if (el) {{ el.textContent = text; el.scrollIntoView({{behavior:"smooth", block:"end"}}); }}
}}

function handleKey(e) {{
  if (e.key === "Enter" && !e.shiftKey) {{ e.preventDefault(); sendMessage(); }}
}}

// ── Context injection: pick transactions relevant to the user's query ─────
const MONTH_MAP = {{
  jan:"01",january:"01",feb:"02",february:"02",mar:"03",march:"03",
  apr:"04",april:"04",may:"05",jun:"06",june:"06",
  jul:"07",july:"07",aug:"08",august:"08",sep:"09",september:"09",
  oct:"10",october:"10",nov:"11",november:"11",dec:"12",december:"12"
}};

function getQueryContext(query) {{
  const q = query.toLowerCase();
  let matched = [];

  // Match category names
  const cats = Object.keys(CAT_COLORS);
  for (const cat of cats) {{
    if (q.includes(cat.toLowerCase())) {{
      matched.push(...TRANSACTIONS.filter(t => t.category === cat));
    }}
  }}

  // Match month names and year-month patterns (e.g. "october", "2025-10")
  for (const [name, num] of Object.entries(MONTH_MAP)) {{
    if (q.includes(name)) {{
      // Try to find a year nearby in the query
      const yearMatch = q.match(/20\d{{2}}/);
      const ym = yearMatch ? `${{yearMatch[0]}}-${{num}}` : `-${{num}}-`;
      matched.push(...TRANSACTIONS.filter(t =>
        yearMatch ? t.date.startsWith(ym) : t.date.includes(ym)
      ));
    }}
  }}
  const ymDirect = q.match(/20\d{{2}}-\d{{2}}/g);
  if (ymDirect) {{
    for (const ym of ymDirect) {{
      matched.push(...TRANSACTIONS.filter(t => t.date.startsWith(ym)));
    }}
  }}

  // Match merchant keywords (words > 3 chars not in stop list)
  const stopWords = new Set(["what","were","when","how","much","did","spend","show","list",
    "the","and","for","with","that","this","have","from","they","been","more","some",
    "than","then","also","your","like","over","last","this","month","year","all"]);
  const words = q.replace(/[^a-z0-9 ]/g, " ").split(/\s+/).filter(w => w.length > 3 && !stopWords.has(w));
  for (const word of words) {{
    matched.push(...TRANSACTIONS.filter(t =>
      t.merchant.toLowerCase().includes(word) && !matched.includes(t)
    ));
  }}

  // Deduplicate by date+merchant+amount
  const seen = new Set();
  const unique = matched.filter(t => {{
    const k = t.date + t.merchant + t.amount;
    if (seen.has(k)) return false;
    seen.add(k); return true;
  }});

  if (!unique.length) return "";

  // Format as a compact table
  const rows = unique
    .sort((a,b) => b.date.localeCompare(a.date))
    .map(t => `  ${{t.date}} | ${{t.merchant.substring(0,42).padEnd(42)}} | ${{t.category.padEnd(14)}} | ${{t.amount<0?"-":"+"}}\$${{Math.abs(t.amount).toFixed(2)}}`)
    .join("\\n");
  return `\\n\\n[RELEVANT TRANSACTIONS (${{unique.length}} matched)]\\n${{rows}}`;
}}

async function sendMessage() {{
  if (isStreaming) return;
  const input = document.getElementById("chatInput");
  const msg   = input.value.trim();
  if (!msg) return;
  input.value = "";
  input.style.height = "56px";

  appendMsg("user", msg);

  // Augment the message with relevant transactions before pushing to history
  const augmented = msg + getQueryContext(msg);
  chatHistory.push({{role:"user", content:augmented}});

  const aid = appendMsg("assistant", "");
  const el  = document.getElementById(aid);
  el.classList.add("streaming");
  isStreaming = true;
  document.getElementById("sendBtn").disabled = true;
  document.getElementById("chatStatus").textContent = "typing…";

  try {{
    const resp = await fetch("http://localhost:11434/api/chat", {{
      method: "POST",
      headers: {{"Content-Type":"application/json"}},
      body: JSON.stringify({{
        model: "puck",
        options: {{num_ctx: 16384}},
        messages: [
          {{role:"system", content: SYSTEM_PROMPT}},
          ...chatHistory
        ],
        stream: true
      }})
    }});

    if (!resp.ok) throw new Error("Ollama returned " + resp.status);

    const reader  = resp.body.getReader();
    const decoder = new TextDecoder();
    let full = "";

    while (true) {{
      const {{done, value}} = await reader.read();
      if (done) break;
      for (const line of decoder.decode(value, {{stream:true}}).split("\\n")) {{
        if (!line.trim()) continue;
        try {{
          const d = JSON.parse(line);
          if (d.message?.content) {{ full += d.message.content; updateMsg(aid, full); }}
          if (d.done) chatHistory.push({{role:"assistant", content:full}});
        }} catch {{}}
      }}
    }}
  }} catch (e) {{
    const errMsg = e.message.includes("fetch") || e.message.includes("Failed")
      ? "⚠ Can't reach Ollama. Make sure it's running: ollama serve"
      : "⚠ " + e.message;
    updateMsg(aid, errMsg);
  }}

  el.classList.remove("streaming");
  isStreaming = false;
  document.getElementById("sendBtn").disabled = false;
  document.getElementById("chatStatus").textContent = "puck · ready";
}}

// ── Transactions table ────────────────────────────────────────────────────
let activeFilter = "All", sortKey = "date", sortAsc = false;

(function buildFilters() {{
  const cats = ["All",...new Set(TRANSACTIONS.map(t=>t.category))].sort((a,b)=>{{
    if(a==="All")return -1; if(b==="All")return 1;
    if(a==="Transfer")return 1; if(b==="Transfer")return -1;
    return a.localeCompare(b);
  }});
  const row = document.getElementById("filterRow");
  cats.forEach(cat=>{{
    const btn=document.createElement("button"); btn.className="fb"+(cat==="All"?" on":"");
    btn.textContent=cat;
    if(cat!=="All") btn.style.setProperty("--cc",cc(cat));
    btn.onclick=()=>{{ activeFilter=cat; document.querySelectorAll(".fb").forEach(b=>b.classList.remove("on")); btn.classList.add("on"); applyFilters(); }};
    row.appendChild(btn);
  }});
}})();

function sortBy(k) {{ if(sortKey===k)sortAsc=!sortAsc; else{{sortKey=k;sortAsc=k==="date"?false:true;}} applyFilters(); }}

function applyFilters() {{
  const q = document.getElementById("txnSearch").value.toLowerCase().trim();
  let rows = TRANSACTIONS.filter(t=>{{
    if(activeFilter!=="All"&&t.category!==activeFilter)return false;
    if(q&&!t.merchant.toLowerCase().includes(q)&&!t.category.toLowerCase().includes(q))return false;
    return true;
  }});
  rows.sort((a,b)=>{{
    let av=a[sortKey],bv=b[sortKey];
    if(typeof av==="string"){{av=av.toLowerCase();bv=bv.toLowerCase();}}
    return (av<bv?-1:av>bv?1:0)*(sortAsc?1:-1);
  }});
  document.getElementById("txnCount").textContent =
    rows.length===TRANSACTIONS.length?`${{TRANSACTIONS.length}} transactions`:`${{rows.length}} of ${{TRANSACTIONS.length}}`;
  const tbody=document.getElementById("txnBody"), noRows=document.getElementById("noRows");
  if(!rows.length){{tbody.innerHTML="";noRows.style.display="block";return;}}
  noRows.style.display="none";
  tbody.innerHTML=rows.map(t=>{{
    const e=t.amount<0;
    return `<tr>
      <td style="white-space:nowrap;color:#4a5568">${{t.date}}</td>
      <td>${{t.merchant}}${{t.dup?'<span class="flag dup">DUP?</span>':''}}${{t.anomaly?'<span class="flag anom">!</span>':''}}</td>
      <td><span class="badge" style="background:${{cc(t.category)}}">${{t.category}}</span></td>
      <td class="${{e?'red':'green'}} num" style="font-variant-numeric:tabular-nums">${{e?'-':'+'}}\$${{Math.abs(t.amount).toFixed(2)}}</td>
    </tr>`;
  }}).join("");
}}
applyFilters();
</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print("  Saved dashboard.html")


def export(results: dict, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    print("Exporting results...")
    export_csvs(results, output_dir)
    generate_html_dashboard(results, output_dir / "dashboard.html")
    print(f"\nAll outputs written to: {output_dir.resolve()}")
