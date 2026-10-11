"""The single shared formatting contract for the compact two-row summary table.

| Field Of Invention | <LABEL> |
| :--- | :--- |
| Classification (IPC) | <display code> — <confidence>; <next display code> — <confidence> |

Built only from validated supported candidates (never alternatives). Markdown export escapes pipes
and newlines. The UI renders the same `classification_lines` semantically (one line per code).
"""

from __future__ import annotations

from app.schemas.classify import Candidate, Confidence, Summary, SummaryLine

INSUFFICIENT = "Insufficient evidence"
DASH = "\u2014"


def confidence_text(c: Confidence) -> str:
    if c.status == "calibrated" and c.score is not None:
        return f"{round(100 * c.score, 1)}%"
    return {
        "not_calibrated": "confidence: not calibrated",
        "incompatible_artifact": "confidence: calibrator incompatible",
        "unsupported_input_mode": "confidence: not calibrated for this input mode",
    }.get(c.status, "confidence: not calibrated")


def md_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()


def build_summary(field_label: str, candidates: list[Candidate]) -> Summary:
    lines = [SummaryLine(display_code=c.display_code, confidence_text=confidence_text(c.confidence)) for c in candidates]
    cell = "; ".join(f"{ln.display_code} {DASH} {ln.confidence_text}" for ln in lines) if lines else INSUFFICIENT
    markdown = (
        f"| Field Of Invention | {md_escape(field_label)} |\n"
        "| :--- | :--- |\n"
        f"| Classification (IPC) | {md_escape(cell)} |\n"
    )
    return Summary(field_of_invention=field_label, classification_lines=lines, classification_cell_text=cell,
                   markdown=markdown)
