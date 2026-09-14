from __future__ import annotations

import html
from datetime import UTC, datetime
from typing import Any

from meerax.report.builder import ReportBuilder, ReportSection

GOVERNANCE_BANNER = (
    "GOVERNANCE NOTICE: This report supports a human reviewer's judgment of translation quality; "
    "it does not autonomously select or deploy a checkpoint. Per-epoch commentary is an "
    "LLM-generated description of the sampled image grid shown alongside it, not an independently "
    "verified quality claim."
)


class EpochReport:
    def __init__(self, epoch: int, losses: dict[str, float], grid_fig, commentary: str) -> None:
        self.epoch = epoch
        self.losses = losses
        self.grid_fig = grid_fig
        self.commentary = commentary


def _run_info_html(run_metadata: dict[str, Any], timestamp: str) -> str:
    fields = ", ".join(f"{k}={v}" for k, v in run_metadata.items())
    return f"<p class='meta'>Run Info &mdash; {html.escape(fields)} | generated: {html.escape(timestamp)}</p>"


def build_report(
    title: str,
    epoch_reports: list[EpochReport],
    run_metadata: dict[str, Any] | None = None,
) -> ReportBuilder:
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    report = ReportBuilder(title, subtitle=GOVERNANCE_BANNER)

    for entry in epoch_reports:
        section = ReportSection(
            title=f"Epoch {entry.epoch}",
            content=html.escape(entry.commentary),
            metrics=entry.losses,
            figures=[entry.grid_fig],
        )
        report.add_section(section)

    report.add_section(
        ReportSection(title="Run Info", content=_run_info_html(run_metadata or {}, timestamp))
    )
    return report
