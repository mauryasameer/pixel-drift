from __future__ import annotations

from forge.report.builder import ReportBuilder, ReportSection


class EpochReport:
    def __init__(self, epoch: int, losses: dict[str, float], grid_fig, commentary: str) -> None:
        self.epoch = epoch
        self.losses = losses
        self.grid_fig = grid_fig
        self.commentary = commentary


def build_report(title: str, epoch_reports: list[EpochReport]) -> ReportBuilder:
    report = ReportBuilder(title)
    for entry in epoch_reports:
        section = ReportSection(
            title=f"Epoch {entry.epoch}",
            content=entry.commentary,
            metrics=entry.losses,
            figures=[entry.grid_fig],
        )
        report.add_section(section)
    return report
