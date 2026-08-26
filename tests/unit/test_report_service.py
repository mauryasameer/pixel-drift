import matplotlib.pyplot as plt

from src.services.report_service import EpochReport, build_report


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
