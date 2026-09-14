import matplotlib.pyplot as plt

from src.services.report_service import GOVERNANCE_BANNER, EpochReport, build_report


def test_build_report_includes_each_epoch_section():
    fig, _ = plt.subplots()
    reports = [
        EpochReport(epoch=5, losses={"gen_g_loss": 1.2}, grid_fig=fig, commentary="improving"),
        EpochReport(epoch=10, losses={"gen_g_loss": 0.9}, grid_fig=fig, commentary="sharper edges"),
    ]

    report = build_report("Test Report", reports)
    html = report.to_html()

    assert "Epoch 5" in html
    assert "Epoch 10" in html
    assert "improving" in html
    assert "sharper edges" in html


def test_build_report_includes_governance_banner_and_run_info():
    fig, _ = plt.subplots()
    reports = [EpochReport(epoch=5, losses={"gen_g_loss": 1.2}, grid_fig=fig, commentary="improving")]

    report = build_report("Test Report", reports, run_metadata={"llm_provider": "ollama", "epochs_trained": 5})
    html = report.to_html()

    assert GOVERNANCE_BANNER in html
    assert "Run Info" in html
    assert "ollama" in html
    assert "epochs_trained=5" in html


def test_build_report_escapes_commentary_content():
    fig, _ = plt.subplots()
    reports = [EpochReport(epoch=5, losses={}, grid_fig=fig, commentary="<script>alert(1)</script>")]

    report = build_report("Test Report", reports)
    html = report.to_html()

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
