from __future__ import annotations

import argparse
import logging
import sys

from meerax.llm.claude import ClaudeProvider
from meerax.llm.ollama import OllamaProvider
from meerax.llm.openai_provider import OpenAIProvider

from src.providers.pix2pix_factory import Pix2PixGeneratorFactory
from src.services.data_service import load_domain
from src.services.narrative_service import generate_commentary
from src.services.report_service import EpochReport, build_report
from src.services.training_service import CycleGANTrainer

logger = logging.getLogger(__name__)

FACTORY_CLS: type = Pix2PixGeneratorFactory

LLM_PROVIDERS: dict[str, type] = {
    "ollama": OllamaProvider,
    "claude": ClaudeProvider,
    "openai": OpenAIProvider,
}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(prog="pixel-drift")
    parser.add_argument("--domain-x-dir", default="src/data/domain_x")
    parser.add_argument("--domain-y-dir", default="src/data/domain_y")
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Square image size images are resized to; must be a multiple of 256 "
        "(the pix2pix U-Net downsamples 8x)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--sample-interval", type=int, default=5)
    parser.add_argument("--llm-provider", choices=list(LLM_PROVIDERS.keys()), default="ollama")
    parser.add_argument("--output", default="reports/model_report.html")
    args = parser.parse_args(argv)

    if args.image_size <= 0 or args.image_size % 256 != 0:
        parser.error("--image-size must be a positive multiple of 256 (the pix2pix U-Net downsamples 8x)")

    try:
        domain_x = load_domain(args.domain_x_dir, image_size=args.image_size)
        domain_y = load_domain(args.domain_y_dir, image_size=args.image_size)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    sample_x = next(iter(domain_x))
    sample_y = next(iter(domain_y))

    trainer = CycleGANTrainer(FACTORY_CLS(), checkpoint_dir=args.checkpoint_dir)
    llm = LLM_PROVIDERS[args.llm_provider]()

    epoch_reports: list[EpochReport] = []

    def on_epoch_end(epoch: int, losses: dict) -> None:
        if epoch % args.sample_interval != 0:
            return
        translated = trainer.generator_g(sample_x, training=False)
        result = generate_commentary(
            sample_x[0].numpy(), translated[0].numpy(), sample_y[0].numpy(), llm
        )
        epoch_reports.append(
            EpochReport(
                epoch=epoch,
                losses={k: float(v) for k, v in losses.items()},
                grid_fig=result.grid_fig,
                commentary=result.commentary,
            )
        )

    epoch_before = int(trainer.checkpoint.epoch)
    trainer.fit(
        domain_x,
        domain_y,
        epochs=args.epochs,
        checkpoint_interval=args.checkpoint_interval,
        on_epoch_end=on_epoch_end,
    )
    epoch_after = int(trainer.checkpoint.epoch)

    if not epoch_reports:
        if epoch_after <= epoch_before:
            logger.warning("no epochs ran (already at or past --epochs count) — report not written")
        else:
            logger.warning(
                "%d epoch(s) ran but none landed on --sample-interval=%d — report not written",
                epoch_after - epoch_before,
                args.sample_interval,
            )
        return 0

    report = build_report("PixelDrift — CycleGAN Training Report", epoch_reports)
    report.save(args.output)
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
