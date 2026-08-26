# pixel-drift — Task Tracker

Live progress tracker. Keep this in sync with actual state.

## Backlog

- [x] Core `GeneratorFactory` interface
- [x] Pix2Pix provider (tensorflow_examples wrapper, grayscale channel adaptation)
- [x] Data service (domain loading, normalization)
- [x] Training service (CycleGAN loop, real checkpointing)
- [x] Narrative service (LLM vision commentary)
- [x] Report service
- [x] CLI driver
- [ ] Real end-to-end run against actual domain_x/domain_y image data (source unrecoverable from the original notebook, not exercised in CI)
- [ ] Real LLM commentary review against actual Ollama/Claude/OpenAI vision output (stubbed in tests)
