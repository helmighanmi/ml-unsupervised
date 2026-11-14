# Path: src/ml_unsupervised/reporting.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-helmi

"""Generate Markdown and HTML reports for unsupervised analysis runs."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data._"
    try:
        return df.to_markdown(index=False)
    except Exception:
        headers = [str(c) for c in df.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
        return "\n".join(lines)


def build_markdown_report(
    *,
    title: str,
    summary: dict[str, Any],
    profile: pd.DataFrame | None = None,
    notes: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Render a structured Markdown report."""
    ts = datetime.now(timezone.utc).isoformat()
    lines: list[str] = [f"# {title}", "", f"_Generated (UTC): {ts}_", "", "## Summary", ""]
    lines.append("```json")
    lines.append(json.dumps(summary, indent=2, sort_keys=True, default=str))
    lines.append("```")
    if metadata:
        lines.extend(
            [
                "",
                "## Metadata",
                "",
                "```json",
                json.dumps(metadata, indent=2, sort_keys=True, default=str),
                "```",
            ]
        )
    if profile is not None:
        lines.extend(["", "## Profile / diagnostics table", "", _df_to_markdown(profile)])
    if notes:
        lines.extend(["", "## Notes & limitations", ""])
        for note in notes:
            lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def markdown_to_html(markdown_text: str, *, title: str = "Report") -> str:
    """Convert a simple Markdown report to standalone HTML (lightweight)."""
    # Minimal conversion sufficient for our generated reports.
    body_lines: list[str] = []
    in_code = False
    for line in markdown_text.splitlines():
        if line.startswith("```"):
            if not in_code:
                body_lines.append("<pre><code>")
                in_code = True
            else:
                body_lines.append("</code></pre>")
                in_code = False
            continue
        if in_code:
            body_lines.append(html.escape(line))
            continue
        if line.startswith("# "):
            body_lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body_lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            body_lines.append(f"<li>{html.escape(line[2:])}</li>")
        elif line.startswith("| "):
            body_lines.append(f"<div class='row'>{html.escape(line)}</div>")
        elif line.strip() == "":
            body_lines.append("<br/>")
        elif line.startswith("_") and line.endswith("_"):
            body_lines.append(f"<p><em>{html.escape(line.strip('_'))}</em></p>")
        else:
            body_lines.append(f"<p>{html.escape(line)}</p>")
    body = "\n".join(body_lines)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>"
        "<style>body{font-family:system-ui,Arial,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;}"
        "pre{background:#f6f8fa;padding:1rem;overflow:auto;} .row{font-family:monospace;white-space:pre;}</style>"
        f"</head><body>{body}</body></html>"
    )


def write_report(
    output: str | Path,
    *,
    title: str,
    summary: dict[str, Any],
    profile: pd.DataFrame | None = None,
    notes: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    format: str | None = None,
) -> Path:
    """Write a report as .md or .html based on suffix/format."""
    path = Path(output)
    fmt = (format or path.suffix.lstrip(".") or "md").lower()
    md = build_markdown_report(title=title, summary=summary, profile=profile, notes=notes, metadata=metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "html":
        path.write_text(markdown_to_html(md, title=title), encoding="utf-8")
    else:
        path.write_text(md, encoding="utf-8")
    return path
