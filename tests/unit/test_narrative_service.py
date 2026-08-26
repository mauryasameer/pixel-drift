import numpy as np
from forge.llm.base import LLMResponse

from src.services.narrative_service import generate_commentary


class _StubLLM:
    def __init__(self, content="a description", raise_error=False):
        self._content = content
        self._raise_error = raise_error
        self.calls = []

    def generate(self, prompt, system=None, images=None, **kwargs):
        self.calls.append({"prompt": prompt, "images": images})
        if self._raise_error:
            raise RuntimeError("LLM unreachable")
        return LLMResponse(content=self._content, model="stub", input_tokens=1, output_tokens=1)

    def chat(self, messages, system=None, **kwargs):
        return self.generate(messages[-1]["content"])


def _tiny_image():
    return np.zeros((8, 8, 1), dtype=np.float32) - 1.0


def test_generate_commentary_passes_grid_image_to_llm():
    llm = _StubLLM(content="looks good")
    result = generate_commentary(_tiny_image(), _tiny_image(), _tiny_image(), llm)

    assert result.commentary == "looks good"
    assert result.grid_fig is not None
    assert len(llm.calls) == 1
    assert llm.calls[0]["images"] is not None
    assert len(llm.calls[0]["images"]) == 1
    assert isinstance(llm.calls[0]["images"][0], bytes)


def test_generate_commentary_falls_back_on_llm_failure():
    llm = _StubLLM(raise_error=True)
    result = generate_commentary(_tiny_image(), _tiny_image(), _tiny_image(), llm)

    assert result.commentary == "commentary unavailable"
    assert result.grid_fig is not None
