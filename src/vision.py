"""Local vision extraction: pytesseract OCR + claude -p structured parsing.

Originally spec'd around Gemini vision (response_schema + inline image
bytes). No working GEMINI_API_KEY was available in Session 1 (two keys
tested, both authenticated but had zero free-tier quota provisioned --
see AI_GAP_PROJECTS_ROADMAP.md's session log), so this was rebuilt as a
fully local pipeline: OCR extracts text, then claude -p (already proven
reliable via the CLI, unlimited on the Max subscription) does the
domain-check + structured extraction from that text -- reusing
guardrails.generate_validated's existing retry/validation loop rather than
inventing a new one.

Two local vision-language models (Qwen2-VL-2B, Qwen2.5-VL-3B via mlx-vlm)
were tried first for the harder case of reading a chart's *shape* (Race Day
Copilot's elevation profiles) rather than just text -- both produced
degenerate output ("The the image.", fragments of axis-label text with no
real curve understanding). Not a setup bug: two different checkpoints, same
failure pattern. Small quantized VLMs are not reliable at this task on this
hardware. Course-chart reading therefore degrades gracefully to the app's
existing human-in-the-loop editable segment table rather than pretending
local vision handles it -- see race-day-copilot/app.py.

Same public API (extract()) as the original Gemini version, so every
project's app.py needed zero changes to its call sites.
"""
from __future__ import annotations

from typing import TypeVar

from PIL import Image
from pydantic import BaseModel, ValidationError, create_model

from guardrails import GuardrailError, Refusal, generate_validated

T = TypeVar("T", bound=BaseModel)

# OCR wants more resolution than the old Gemini-token-cost-driven cap (1568px)
# -- there's no per-token cost locally, and tesseract accuracy degrades on
# small text if the image is downscaled too aggressively.
_MAX_OCR_LONG_SIDE = 2400


def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    img = image.convert("L")  # grayscale measurably helps tesseract on photos
    w, h = img.size
    longest = max(w, h)
    if longest > _MAX_OCR_LONG_SIDE:
        scale = _MAX_OCR_LONG_SIDE / longest
        img = img.resize((round(w * scale), round(h * scale)))
    return img


def ocr_text(image: Image.Image, langs: str = "eng") -> str:
    """langs: tesseract language code(s), '+'-joined for multi-language
    documents (e.g. "eng+jpn+tha" for a menu of unknown language). Available
    packs as of Session 1: eng, jpn, tha, ita, chi_sim, chi_tra, kor, vie,
    spa, fra, deu, por, ell (see _ai-gap-toolkit's setup notes)."""
    import pytesseract
    return pytesseract.image_to_string(preprocess_for_ocr(image), lang=langs)


def _envelope_schema(schema: type[T]) -> type[BaseModel]:
    return create_model(
        f"{schema.__name__}Envelope",
        is_target_domain=(bool, ...),
        domain_confidence=(float, ...),
        refusal_reason=(str | None, None),
        data=(schema | None, None),
    )


def extract(image: Image.Image, schema: type[T], task_prompt: str, *,
            domain_description: str, min_confidence: float = 0.6,
            ocr_langs: str = "eng") -> T | Refusal:
    text = ocr_text(image, langs=ocr_langs).strip()

    if len(text) < 5:
        return Refusal(
            reason="OCR extracted no readable text",
            user_message="Sorry, I couldn't read any text in that image -- try a clearer, well-lit, closer photo.",
            confidence=0.0,
        )

    envelope_schema = _envelope_schema(schema)
    prompt = (
        f"The following text was OCR'd from a photo that should be {domain_description}. "
        "OCR introduces noise -- garbled words, dropped characters, merged lines -- "
        "use your judgment to work around it.\n\n"
        f"OCR'd text:\n\"\"\"\n{text[:4000]}\n\"\"\"\n\n"
        f"First decide whether this text is actually consistent with being "
        f"{domain_description} (is_target_domain, domain_confidence 0-1; if not, set "
        f"refusal_reason to a short user-facing explanation and leave data null). "
        f"If it is, extract: {task_prompt}"
    )

    try:
        envelope = generate_validated(prompt, envelope_schema, max_retries=1)
    except GuardrailError as e:
        return Refusal(
            reason=f"extraction failed after retries: {e.violations}",
            user_message="Sorry, I had trouble reading that clearly -- try a clearer photo, or enter the details manually.",
            confidence=0.0,
        )

    if not envelope.is_target_domain or envelope.domain_confidence < min_confidence or envelope.data is None:
        return Refusal(
            reason=envelope.refusal_reason or "not in target domain",
            user_message=envelope.refusal_reason or "That doesn't look right -- try a different photo.",
            confidence=envelope.domain_confidence,
        )

    return envelope.data
