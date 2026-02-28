"""
LUMINARK Report Generator
Produces plain-text, Markdown, and CSV audit reports from GuardianResult objects.
"""

from __future__ import annotations

import csv
import io
from typing import List

from .guardian import GuardianResult
from .principles import CATEGORIES


BADGE_EMOJI = {
    "PASS":     "✅",
    "CAUTION":  "⚠️",
    "FAIL":     "❌",
    "CRITICAL": "🚨",
}

DEFENSE_LABELS = {
    "none":       "None Required",
    "monitor":    "Passive Monitor",
    "camouflage": "Octo-Camouflage",
    "contain":    "Mycelial Containment",
    "harrowing":  "Full Harrowing",
    "quarantine": "Quarantine",
}


def generate_text_report(result: GuardianResult) -> str:
    """Plain-text audit report (for CLI output, email, export)."""
    lines = [
        "=" * 70,
        "  LUMINARK ETHICAL AI GUARDIAN — AUDIT REPORT",
        "=" * 70,
        f"  Input ID   : {result.input_id}",
        f"  Timestamp  : {result.timestamp}",
        "",
        f"  BADGE      : {result.badge}",
        f"  Alignment  : {result.alignment_score:.1f} / 100",
        f"  RISS Threat: {result.threat_score:.3f}  (0 = safe, 1 = critical)",
        f"  Confidence : {result.confidence:.0%}",
        f"  SAP Stage  : {result.stage.value} — {result.stage.label}",
        f"               {result.stage.description}",
        f"  Defense    : {DEFENSE_LABELS.get(result.defense.value, result.defense.value)}",
        "",
        "─" * 70,
        "  INPUT TEXT",
        "─" * 70,
        f"  {result.input_text[:500]}{'...' if len(result.input_text) > 500 else ''}",
        "",
    ]

    # Violations
    if result.violations:
        lines += [
            "─" * 70,
            f"  VIOLATIONS ({result.violation_count} found)",
            "─" * 70,
        ]
        for i, v in enumerate(result.violations, 1):
            cat_label = CATEGORIES.get(v.category, v.category)
            lines += [
                f"  [{i}] {v.label}  (severity: {v.severity:.0%})",
                f"      Category : {cat_label}",
                f"      Issue    : {v.description}",
                f"      Triggers : {', '.join(v.matched_terms[:5])}",
                "",
            ]
    else:
        lines += [
            "─" * 70,
            "  VIOLATIONS: None detected",
            "",
        ]

    # Category breakdown
    if result.category_breakdown:
        lines += ["─" * 70, "  CATEGORY BREAKDOWN", "─" * 70]
        for cat, count in sorted(result.category_breakdown.items(), key=lambda x: -x[1]):
            cat_label = CATEGORIES.get(cat, cat)
            lines.append(f"  {cat_label:<28} {count} violation(s)")
        lines.append("")

    # Bio-defenses
    lines += [
        "─" * 70,
        "  BIO-DEFENSE STATUS",
        "─" * 70,
        f"  {result.containment_msg}",
        f"  {result.camouflage_msg}",
        "",
    ]

    # Sentence-level
    if len(result.sentence_analyses) > 1:
        lines += ["─" * 70, "  SENTENCE-LEVEL ANALYSIS", "─" * 70]
        for i, sa in enumerate(result.sentence_analyses, 1):
            flag = "⚠" if sa.alignment < 0.65 else "✓"
            vcount = len(sa.violations)
            lines.append(
                f"  [{i}] {flag} Align={sa.alignment:.0%}  Violations={vcount}  "
                f"— {sa.text[:80]}{'...' if len(sa.text) > 80 else ''}"
            )
        lines.append("")

    # Rewrite
    lines += [
        "─" * 70,
        "  COMPASSIONATE REWRITE",
        "─" * 70,
    ]
    if result.rewrite_applied:
        lines += [
            f"  Changes applied: {len(result.changes_made)}",
            "",
        ]
        for change in result.changes_made:
            lines.append(f"    • {change}")
        lines += ["", "  Rewritten text:", f"  {result.rewrite}"]
    else:
        lines += ["  No rewrite needed — output is well-calibrated.", f"  {result.rewrite}"]

    # Recommendations
    lines += [
        "",
        "─" * 70,
        "  RECOMMENDATIONS",
        "─" * 70,
    ]
    for rec in result.recommendations:
        lines.append(f"  → {rec}")

    lines += [
        "",
        "=" * 70,
        "  Powered by LUMINARK Ethical AI Guardian v1.0",
        "  Bio-inspired • Ma'at-audited • Compassionate AI safety",
        "=" * 70,
    ]
    return "\n".join(lines)


def generate_markdown_report(result: GuardianResult) -> str:
    """Markdown report suitable for GitHub, Notion, or web display."""
    badge_emoji = BADGE_EMOJI.get(result.badge, "")
    lines = [
        "# LUMINARK Ethical AI Guardian — Audit Report",
        "",
        f"| Field | Value |",
        f"|---|---|",
        f"| **Badge** | {badge_emoji} **{result.badge}** |",
        f"| Alignment Score | {result.alignment_score:.1f} / 100 |",
        f"| RISS Threat Score | {result.threat_score:.3f} |",
        f"| Confidence | {result.confidence:.0%} |",
        f"| SAP Stage | {result.stage.value} — {result.stage.label} |",
        f"| Defense Mode | {DEFENSE_LABELS.get(result.defense.value, result.defense.value)} |",
        f"| Input ID | `{result.input_id}` |",
        f"| Timestamp | `{result.timestamp}` |",
        "",
        "## Input",
        "",
        f"> {result.input_text[:600]}{'...' if len(result.input_text) > 600 else ''}",
        "",
    ]

    # Violations
    if result.violations:
        lines += [f"## Violations ({result.violation_count})", ""]
        for v in result.violations:
            cat_label = CATEGORIES.get(v.category, v.category)
            sev_bar = "█" * int(v.severity * 10) + "░" * (10 - int(v.severity * 10))
            lines += [
                f"### ⚠ {v.label}",
                f"- **Category**: {cat_label}",
                f"- **Severity**: `{sev_bar}` {v.severity:.0%}",
                f"- **Issue**: {v.description}",
                f"- **Triggers found**: `{', '.join(v.matched_terms[:5])}`",
                "",
            ]
    else:
        lines += ["## Violations", "", "✅ No violations detected.", ""]

    # Bio-defenses
    lines += [
        "## Bio-Defense Status",
        "",
        f"- {result.containment_msg}",
        f"- {result.camouflage_msg}",
        "",
    ]

    # Rewrite
    lines += ["## Compassionate Rewrite", ""]
    if result.rewrite_applied:
        lines += [
            f"**{len(result.changes_made)} changes applied:**",
            "",
        ]
        for change in result.changes_made:
            lines.append(f"- {change}")
        lines += [
            "",
            "**Rewritten output:**",
            "",
            f"```\n{result.rewrite}\n```",
        ]
    else:
        lines += [
            "_No rewrite needed — output is well-calibrated._",
            "",
            f"```\n{result.rewrite}\n```",
        ]

    # Recommendations
    lines += ["", "## Recommendations", ""]
    for rec in result.recommendations:
        lines.append(f"- {rec}")

    lines += [
        "",
        "---",
        "_Powered by [LUMINARK Ethical AI Guardian](https://github.com/luminark/guardian) v1.0_",
    ]
    return "\n".join(lines)


def generate_csv_row(result: GuardianResult) -> dict:
    """Single row dict for CSV batch export."""
    return {
        "input_id":         result.input_id,
        "timestamp":        result.timestamp,
        "badge":            result.badge,
        "alignment_score":  round(result.alignment_score, 1),
        "threat_score":     round(result.threat_score, 3),
        "confidence":       round(result.confidence, 3),
        "stage":            result.stage.name,
        "stage_label":      result.stage.label,
        "defense_mode":     result.defense.value,
        "violation_count":  result.violation_count,
        "violations":       "; ".join(v.label for v in result.violations),
        "categories_hit":   "; ".join(sorted(result.category_breakdown.keys())),
        "rewrite_applied":  result.rewrite_applied,
        "changes_count":    len(result.changes_made),
        "input_preview":    result.input_text[:200].replace("\n", " "),
        "rewrite_preview":  result.rewrite[:200].replace("\n", " "),
        "top_recommendation": result.recommendations[0] if result.recommendations else "",
    }


def batch_to_csv(results: List[GuardianResult]) -> str:
    """Convert a list of results to a CSV string."""
    if not results:
        return ""
    output = io.StringIO()
    rows = [generate_csv_row(r) for r in results]
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()
