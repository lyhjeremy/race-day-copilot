"""STT (faster-whisper) + TTS (edge-tts) helpers, with file-hash caching for
synthesized audio.

Env: AUDIO_STT_SIZE overrides the whisper model size (default: "base" on HF
Spaces via SPACE_ID, "small" locally).
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

from cache import FileCache

VOICES = {
    "en": "en-US-AriaNeural",
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
    "it": "it-IT-ElsaNeural",
    "de": "de-DE-KatjaNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "th": "th-TH-PremwadeeNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "pt": "pt-BR-FranciscaNeural",
    "vi": "vi-VN-HoaiMyNeural",
    "el": "el-GR-AthinaNeural",
}


class NoSpeechError(RuntimeError):
    pass


@dataclass
class Transcript:
    text: str
    language: str
    avg_logprob: float
    no_speech_prob: float
    duration_s: float


_stt_model = None


def _stt_model_size() -> str:
    override = os.environ.get("AUDIO_STT_SIZE")
    if override:
        return override
    return "base" if os.environ.get("SPACE_ID") else "small"


def _get_stt_model():
    global _stt_model
    if _stt_model is None:
        from faster_whisper import WhisperModel
        _stt_model = WhisperModel(_stt_model_size(), device="cpu", compute_type="int8")
    return _stt_model


def transcribe(audio_path: str | Path, language: str | None = None) -> Transcript:
    model = _get_stt_model()
    segments, info = model.transcribe(str(audio_path), language=language, vad_filter=True)
    segments = list(segments)
    if not segments:
        raise NoSpeechError("No speech detected in audio.")

    text = " ".join(s.text.strip() for s in segments)
    avg_logprob = sum(s.avg_logprob for s in segments) / len(segments)
    no_speech_prob = max(s.no_speech_prob for s in segments)

    if no_speech_prob > 0.5:
        raise NoSpeechError(f"Low-confidence speech detection (no_speech_prob={no_speech_prob:.2f}).")

    return Transcript(
        text=text, language=info.language, avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob, duration_s=info.duration,
    )


def voice_for(language: str) -> str | None:
    return VOICES.get(language.lower()[:2])


def speak(text: str, voice: str, out_path: str | Path) -> Path:
    """Synthesize `text` in `voice` to `out_path` (sync wrapper over edge-tts)."""
    import edge_tts

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    async def _run():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(out_path))

    asyncio.run(_run())
    return out_path


def speak_cached(text: str, voice: str, cache: FileCache) -> Path:
    """speak() with content-hash caching via a shared FileCache instance."""
    key = f"{voice}:{text}"
    hit = cache.get(key, ".mp3")
    if hit:
        return hit
    out_path = cache.path_for(key, ".mp3")
    return speak(text, voice, out_path)
