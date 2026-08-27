from __future__ import annotations

import io
import logging

import matplotlib.pyplot as plt
import numpy as np
from meerax.llm.base import LLMProvider
from meerax.vision.gridplot import plot_translation_grid

logger = logging.getLogger(__name__)

COMMENTARY_PROMPT = (
    "This image shows a CycleGAN translation sample: input image, translated output, and "
    "reference image side by side. In 2-3 sentences, describe the translation quality — "
    "artifacts, texture transfer, and how close the translation looks to the reference."
)


class CommentaryResult:
    def __init__(self, commentary: str, grid_fig: plt.Figure) -> None:
        self.commentary = commentary
        self.grid_fig = grid_fig


def _figure_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    return buf.getvalue()


def generate_commentary(
    input_image: np.ndarray,
    translated_image: np.ndarray,
    reference_image: np.ndarray,
    llm: LLMProvider,
) -> CommentaryResult:
    fig = plot_translation_grid(
        [
            ("Input", input_image),
            ("Translated", translated_image),
            ("Reference", reference_image),
        ]
    )
    try:
        png_bytes = _figure_to_png_bytes(fig)
        response = llm.generate(COMMENTARY_PROMPT, images=[png_bytes])
        commentary = response.content
    except Exception:
        logger.exception("Narrative commentary generation failed")
        commentary = "commentary unavailable"
    return CommentaryResult(commentary=commentary, grid_fig=fig)
