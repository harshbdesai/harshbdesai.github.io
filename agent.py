"""
Data Quality Monitoring Agent
Uses Claude (Anthropic API) to autonomously scan a dataset for anomalies,
generate structured findings, and produce a Markdown report.
"""

import json
import os
import re
from datetime import datetime
from io import StringIO

import anthropic
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096
SAMPLE_ROWS = 5          # rows sent to Claude for context
TOP_N_ISSUES = 20        # max flagged rows included in the report

# ── Tools the agent can call ──────────────────────────────────────────────────
TOOLS = [
    {
        "name": "compute_column_stats",
        "description": (
            "Compute descriptive statistics and null counts for one or more "
            "columns in the dataset. Returns min, max, mean, std, null_count, "
            "unique_count, and sample values."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of column names to analyse.",
                }
            },
            "required": ["columns"],
        },
    },
    {
        "name": "flag_anomalies",
        "description": (
            "Flag rows that look anomalous for a given column based on a "
            "rule. Supported rules: 'zscore' (|z| > threshold), 'null', "
            "'negative', 'out_of_range' (requires min_val / max_val)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "column":    {"type": "string",  "description": "Column to check."},
                "rule":      {"type": "string",  "enum": ["zscore", "null", "negative", "out_of_range"]},
                "threshold": {"type": "number",  "description": "Z-score threshold (default 3)."},
                "min_val":   {"type": "number",  "description": "Min value for out_of_range rule."},
                "max_val":   {"type": "number",  "description": "Max value for out_of_range rule."},
            },
            "required": ["column", "rule"],
        },
    },
    {
        "name": "check_duplicates",
        "description": "Check for duplicate rows across the whole dataset or a subset of columns.",
        "input_schema": {
            "type": "object",
            "properties": {
                "subset": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columns to consider. Omit to check all columns.",
                }
            },
        },
    },
    {
        "name": "finish",
        "description": (
            "Signal that investigation is complete and provide the final "
            "structured findings as JSON."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary":  {"type": "string",  "description": "1-2 sentence plain-English summary."},
                "severity": {"type": "string",  "enum": ["low", "medium", "high"],
                             "description": "Overall data quality severity."},
                "findings": {
                    "type": "array",
                    "description": "List of individual findings.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column":      {"type": "string"},
                            "issue_type":  {"type": "string"},
                            "count":       {"type": "integer"},
                            "detail":      {"type": "string"},
                            "row_indices": {
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                        },
                        "required": ["column", "issue_type", "count", "detail"],
                    },
                },
                "recommendations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Actionable next steps.",
                },
            },
            "required": ["summary", "severity", "findings", "recommendations"],
        },
    },
]

# ── Tool implementations ───────────────────────────────────────────────────────

def compute_column_stats(df: pd.DataFrame, columns: list[str]) -> dict:
    result = {}
    for col in columns:
        if col not in df.columns:
            result[col] = {"error": f"Column '{col}' not found."}
            continue
        s = df[col]
        entry: dict = {
            "dtype":        str(s.dtype),
            "null_count":   int(s.isna().sum()),
            "unique_count": int(s.nunique()),
            "sample":       s.dropna().head(5).tolist(),
        }
        if pd.api.types.is_numeric_dtype(s):
            entry.update({
                "min":  round(float(s.min()), 4) if not s.empty else None,
                "max":  round(float(s.max()), 4) if not s.empty else None,
                "mean": round(float(s.mean()), 4) if not s.empty else None,
                "std":  round(float(s.std()), 4)  if not s.empty else None,
            })
        result[col] = entry
    return result


def flag_anomalies(
    df: pd.DataFrame,
    column: str,
    rule: str,
    threshold: float = 3.0,
    min_val: float | None = None,
    max_val: float | None = None,
) -> dict:
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}
    s = df[column]
    if rule == "null":
        mask = s.isna()
    elif rule == "negative":
        mask = s < 0
    elif rule == "zscore":
        z = (s - s.mean()) / s.std()
        mask = z.abs() > threshold
    elif rule == "out_of_range":
        mask = pd.Series([False] * len(s), index=s.index)
        if min_val is not None:
            mask |= s < min_val
        if max_val is not None:
            mask |= s > max_val
    else:
        return {"error": f"Unknown rule '{rule}'."}

    flagged = df[mask]
    return {
        "column":      column,
        "rule":        rule,
        "flagged_count": int(mask.sum()),
        "flagged_indices": flagged.index[:TOP_N_ISSUES].tolist(),
        "sample_rows": flagged.head(5).to_dict(orient="records"),
    }


def check_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> dict:
    dupes = df[df.duplicated(subset=subset, keep=False)]
    return {
        "duplicate_row_count": int(len(dupes)),
        "duplicate_indices":   dupes.index[:TOP_N_ISSUES].tolist(),
        "sample_rows":         dupes.head(5).to_dict(orient="records"),
    }


def dispatch_tool(df: pd.DataFrame, tool_name: str, tool_input: dict) -> str:
    if tool_name == "compute_column_stats":
        result = compute_column_stats(df, tool_input["columns"])
    elif tool_name == "flag_anomalies":
        result = flag_anomalies(df, **tool_input)
    elif tool_name == "check_duplicates":
        result = check_duplicates(df, tool_input.get("subset"))
    else:
        result = {"error": f"Unknown tool '{tool_name}'."}
    return json.dumps(result, default=str)


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_agent(df: pd.DataFrame, verbose: bool = True) -> dict:
    """Run the agentic loop. Returns the final findings dict."""
    client = anthropic.Anthropic()

    schema_buf = StringIO()
    df.info(buf=schema_buf)
    schema_str = schema_buf.getvalue()
    sample_str = df.head(SAMPLE_ROWS).to_markdown(index=True)

    system_prompt = (
        "You are a data quality monitoring agent. "
        "Your job is to autonomously investigate a dataset for anomalies, "
        "missing values, duplicates, and out-of-range values. "
        "Use the tools available to gather evidence, then call 'finish' with "
        "your structured findings. Be thorough but efficient — aim for 4-8 "
        "tool calls before finishing."
    )

    user_message = (
        f"Please analyse this dataset for data quality issues.\n\n"
        f"**Schema:**\n```\n{schema_str}\n```\n\n"
        f"**Sample rows (first {SAMPLE_ROWS}):**\n{sample_str}\n\n"
        f"Total rows: {len(df)}, Total columns: {len(df.columns)}\n\n"
        "Investigate all columns systematically and call 'finish' when done."
    )

    messages = [{"role": "user", "content": user_message}]
    final_findings: dict = {}
    iteration = 0

    while True:
        iteration += 1
        if verbose:
            print(f"\n{'─'*50}\n🔄  Agent iteration {iteration}")

        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            tools=TOOLS,
            messages=messages,
        )

        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})

        # Collect tool uses from this turn
        tool_uses = [b for b in response.content if b.type == "tool_use"]

        if not tool_uses:
            # No tool calls — agent is done (shouldn't normally happen)
            if verbose:
                print("⚠️  No tool calls in response. Stopping.")
            break

        tool_results = []
        for tu in tool_uses:
            if verbose:
                print(f"  🔧 Tool call: {tu.name}({json.dumps(tu.input, default=str)[:120]}…)")

            if tu.name == "finish":
                final_findings = tu.input
                if verbose:
                    print("  ✅ Agent called 'finish'. Investigation complete.")
                # Return the tool result so Claude can produce a final text response
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps({"status": "report_generated"}),
                })
                messages.append({"role": "user", "content": tool_results})
                return final_findings

            result_str = dispatch_tool(df, tu.name, tu.input)
            if verbose:
                print(f"     → result: {result_str[:200]}…")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": result_str,
            })

        messages.append({"role": "user", "content": tool_results})

        # Safety valve
        if iteration >= 20:
            if verbose:
                print("⚠️  Max iterations reached.")
            break

    return final_findings


# ── Markdown report renderer ──────────────────────────────────────────────────

def render_markdown_report(findings: dict, dataset_name: str, df: pd.DataFrame) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    severity_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(
        findings.get("severity", "low"), "⚪"
    )

    lines = [
        f"# Data Quality Report — {dataset_name}",
        f"_Generated: {now}_",
        "",
        "---",
        "",
        "## Overview",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Dataset | `{dataset_name}` |",
        f"| Rows | {len(df):,} |",
        f"| Columns | {len(df.columns)} |",
        f"| Overall Severity | {severity_emoji} **{findings.get('severity', 'N/A').upper()}** |",
        "",
        f"> {findings.get('summary', '')}",
        "",
        "---",
        "",
        "## Findings",
        "",
    ]

    for i, f in enumerate(findings.get("findings", []), 1):
        lines += [
            f"### {i}. {f.get('issue_type', 'Issue')} — `{f.get('column', '?')}`",
            "",
            f"- **Affected rows:** {f.get('count', 0):,}",
            f"- **Detail:** {f.get('detail', '')}",
        ]
        indices = f.get("row_indices", [])
        if indices:
            lines.append(f"- **Row indices:** {indices[:10]}")
        lines.append("")

    lines += [
        "---",
        "",
        "## Recommendations",
        "",
    ]
    for rec in findings.get("recommendations", []):
        lines.append(f"- {rec}")

    lines += [
        "",
        "---",
        "",
        "## Column Summary",
        "",
        df.describe(include="all").to_markdown(),
        "",
        "---",
        "_Report generated by Data Quality Monitoring Agent — Harsh Desai_",
    ]

    return "\n".join(lines)


# ── Sample dataset factory ────────────────────────────────────────────────────

def make_sample_dataset() -> pd.DataFrame:
    """
    Generic sales/transactions dataset with intentional quality issues:
    - Missing values
    - Negative amounts
    - Outliers
    - Duplicate rows
    """
    import numpy as np
    rng = np.random.default_rng(42)
    n = 200

    data = {
        "transaction_id": list(range(1000, 1000 + n)),
        "customer_id":    rng.integers(1, 50, size=n).tolist(),
        "product_sku":    rng.choice(["SKU-A", "SKU-B", "SKU-C", "SKU-D", None], size=n).tolist(),
        "quantity":       rng.integers(1, 20, size=n).tolist(),
        "unit_price":     (rng.normal(50, 15, size=n)).round(2).tolist(),
        "discount_pct":   (rng.uniform(0, 0.4, size=n)).round(3).tolist(),
        "region":         rng.choice(["EMEA", "APAC", "AMER", None], size=n, p=[0.4, 0.3, 0.25, 0.05]).tolist(),
        "transaction_date": pd.date_range("2024-01-01", periods=n, freq="2h").astype(str).tolist(),
    }

    df = pd.DataFrame(data)

    # Inject issues
    # 1. Negative unit prices
    df.loc[rng.choice(n, 6, replace=False), "unit_price"] = rng.uniform(-30, -1, 6).round(2)
    # 2. Outlier quantities
    df.loc[rng.choice(n, 4, replace=False), "quantity"] = rng.integers(500, 1000, 4)
    # 3. Missing customer_ids
    df.loc[rng.choice(n, 8, replace=False), "customer_id"] = None
    # 4. Discount > 1 (impossible percentage)
    df.loc[rng.choice(n, 5, replace=False), "discount_pct"] = rng.uniform(1.1, 2.0, 5).round(3)
    # 5. Duplicate rows
    dupes = df.sample(5, random_state=1)
    df = pd.concat([df, dupes], ignore_index=True)

    return df


# ── Entry point ───────────────────────────────────────────────────────────────

def main(csv_path: str | None = None, output_path: str = "report.md"):
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        dataset_name = os.path.basename(csv_path)
        print(f"📂 Loaded dataset: {csv_path} ({len(df)} rows, {len(df.columns)} cols)")
    else:
        print("📊 No CSV provided — using built-in sample dataset.")
        df = make_sample_dataset()
        dataset_name = "sample_transactions.csv"

    print(f"\n🤖 Starting Data Quality Agent on '{dataset_name}'…")
    findings = run_agent(df, verbose=True)

    if not findings:
        print("⚠️  Agent returned no findings.")
        return

    report_md = render_markdown_report(findings, dataset_name, df)

    with open(output_path, "w") as f:
        f.write(report_md)

    print(f"\n📄 Report saved → {output_path}")
    print(f"   Severity : {findings.get('severity', '?').upper()}")
    print(f"   Findings : {len(findings.get('findings', []))}")
    print(f"   Recommendations: {len(findings.get('recommendations', []))}")


if __name__ == "__main__":
    import sys
    csv_file = sys.argv[1] if len(sys.argv) > 1 else None
    out_file = sys.argv[2] if len(sys.argv) > 2 else "report.md"
    main(csv_file, out_file)
