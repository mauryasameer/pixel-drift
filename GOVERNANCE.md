# GOVERNANCE.md

## Intended Use

PixelDrift supports a human reviewer's judgment of CycleGAN translation quality across training
— it does not autonomously select, promote, or deploy a checkpoint. Every training report is
decision support for a person deciding whether a given epoch's translation is good enough to
use, not an autonomous approval step.

## Explainability Boundary

Per-epoch commentary is an LLM-generated description of the sampled input/translated/reference
image grid shown alongside it in the report — it describes what the images in that grid look
like (artifacts, texture transfer, closeness to the reference), never a claim about the model's
internal reasoning or a guarantee about generalization beyond the sampled images shown. This
boundary is stated in the report itself (a permanent banner), not only here.

## Fairness

PixelDrift translates unpaired grayscale image domains (e.g. photo-to-sketch style translation);
it does not make decisions about individuals and the training data carries no protected-attribute
information. A disparate-impact fairness analysis does not apply to this domain — no fairness
proxy is reported here because none would be meaningful for this kind of model.

## LLM Controls

The commentary-generation call passes `temperature=0` to the configured LLM provider, so a given
epoch's commentary is reproducible rather than randomly sampled. The only inputs reaching the
prompt are the fixed instruction text and the sampled image grid itself (input/translated/
reference images) — no retrieved documents or user-supplied free text ever reach it, so the
prompt-injection surface is assessed as low. Multimodal image input is itself a broader surface
than pure text, but the images are the pipeline's own training samples, not externally supplied.

## Audit Trail

Every generated report embeds a "Run Info" block recording the LLM provider used, the number of
epochs trained, the image size, and the checkpoint directory, plus a timestamp — so any given
report is traceable to what produced it.

## Regulatory Framing

PixelDrift is an image-translation research/demo tool, not a financial-services, healthcare, or
HR system — none of SR 11-7, the EU AI Act's high-risk categories, or GDPR's special-category-data
provisions apply to this domain. Noted here for completeness: this project does not currently
operate under a specific regulatory framework, and that absence is itself worth stating rather
than leaving unaddressed.
