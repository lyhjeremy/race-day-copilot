"""Race Day Copilot -- Gradio app. Runs locally and as an HF Space.

Photo of a course elevation profile + spoken/typed goal -> a verified,
grounded pacing plan read back via TTS. See RACE_DAY_COPILOT_SPEC.md §5.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradio as gr

import audio
import vision
from cache import FileCache, SemanticCache
from copilot import assemble_prompt, build_retrieval_query
from guardrails import GuardrailError, Refusal, generate_validated
from schemas import CourseProfile, PacingPlan, RunnerBrief, Segment
from verifier import verify

DATA_DIR = Path(__file__).resolve().parent / "data"
semantic_cache = SemanticCache(DATA_DIR / "plan_cache.db", similarity_threshold=0.93)
audio_cache = FileCache(DATA_DIR / "audio_cache")

EXAMPLE_COURSES_DIR = Path(__file__).resolve().parent / "assets" / "example_courses"


def _course_cache_key(course: CourseProfile, brief: RunnerBrief) -> str:
    """Round goal to nearest 5min and temp to nearest 2C so near-identical
    requests hit the semantic/exact cache -- the token-optimization story."""
    seg_shape = tuple((round(s.km_start), round(s.km_end), s.trend, s.severity) for s in course.segments)
    goal_band = round(brief.goal_time_min / 5) * 5
    temp_band = round((brief.temp_c or 15) / 2) * 2
    return f"{course.distance_km}:{seg_shape}:{goal_band}:{temp_band}:{brief.experience}"


def extract_course(image) -> tuple[CourseProfile | None, str, list[list]]:
    """Returns (course_or_none, status_message, editable_segments_table_rows)."""
    if image is None:
        return None, "Upload or select an example course image.", []

    from PIL import Image
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    result = vision.extract(
        image, CourseProfile,
        task_prompt="race_name if visible, distance_km, and a list of segments "
                    "(km_start, km_end, trend: up/down/flat, severity: gentle/moderate/steep, "
                    "optional note like a hill's name)",
        domain_description="a marathon or half-marathon course elevation profile chart",
        min_confidence=0.55,
    )

    if isinstance(result, Refusal):
        return None, f"⚠ {result.user_message}", []

    rows = [[s.km_start, s.km_end, s.trend, s.severity, s.note or ""] for s in result.segments]
    # source_confidence is the schema's own self-reported read quality --
    # distinct from the domain-gate check above (which only confirms "this is
    # a course chart", not "these specific segments are accurate"). Local
    # OCR+text-only extraction genuinely cannot read a curve's shape (verified
    # live: a hilly course chart image produced a plausible-looking but
    # basically guessed segment list at confidence 0.32, going only off the
    # chart's title text) -- so a low score here isn't a rare edge case for
    # this project, it's the common case, and needs to read as a real warning,
    # not a small number tucked into a sentence.
    if result.source_confidence < 0.5:
        status = (f"⚠ Low-confidence read ({result.source_confidence:.2f}) for {result.race_name or 'this course'}, "
                  f"{result.distance_km}km — local vision can't reliably read a chart's shape, only its text "
                  f"labels. The segments below are a rough guess; please correct them (or use an example course) "
                  f"before building a plan.")
    else:
        status = (f"✓ Extracted: {result.race_name or 'course'}, {result.distance_km}km "
                  f"(confidence {result.source_confidence:.2f}). Review/edit the segments below before continuing.")
    return result, status, rows


def load_example(name: str):
    path = EXAMPLE_COURSES_DIR / f"{name}.json"
    course = CourseProfile.model_validate_json(path.read_text())
    rows = [[s.km_start, s.km_end, s.trend, s.severity, s.note or ""] for s in course.segments]
    img_path = str(EXAMPLE_COURSES_DIR / f"{name}.png")
    return img_path, course, f"✓ Loaded example: {course.race_name}, {course.distance_km}km", rows


def transcribe_goal(audio_path: str | None) -> str:
    if audio_path is None:
        return ""
    try:
        t = audio.transcribe(audio_path)
        return t.text
    except audio.NoSpeechError:
        return ""


def build_plan(course: CourseProfile | None, segments_table, distance_km: float,
               goal_text: str, temp_c: float | None, experience: str, units: str):
    if course is None:
        return "⚠ Extract or select a course first.", None, None, ""

    # rebuild course from the (possibly user-edited) segments table
    segments = [
        Segment(km_start=float(r[0]), km_end=float(r[1]), trend=r[2], severity=r[3], note=r[4] or None)
        for r in segments_table
    ]
    course = CourseProfile(race_name=course.race_name, distance_km=distance_km,
                            segments=segments, source_confidence=course.source_confidence)

    # parse a goal like "sub 4 hours" / "3:45" / "225 minutes" -- lenient
    goal_min = _parse_goal_minutes(goal_text)
    if goal_min is None:
        return f"⚠ Couldn't parse a goal time from '{goal_text}'. Try e.g. '3:45' or '225 minutes'.", None, None, ""

    brief = RunnerBrief(goal_time_min=goal_min, temp_c=temp_c, experience=experience or None, units=units)

    cache_key = _course_cache_key(course, brief)
    prompt, packed, retrieved_ids = assemble_prompt(course, brief)
    cache_hit = semantic_cache.get(prompt, cache_key)
    if cache_hit:
        plan = PacingPlan.model_validate_json(cache_hit.response)
        cached_note = f" (cached, {cache_hit.kind} match, sim={cache_hit.similarity:.2f})"
    else:
        try:
            plan = generate_validated(
                prompt, PacingPlan,
                verifier=lambda p: verify(p, course, brief, retrieved_ids=retrieved_ids),
                max_retries=2, llm_kwargs={"tier": "smart"},
            )
        except GuardrailError as e:
            return f"⚠ Couldn't produce a verified plan: {'; '.join(e.violations[:3])}", None, None, ""
        semantic_cache.put(prompt, cache_key, plan.model_dump_json())
        cached_note = ""

    splits_rows = [[s.km, s.target_pace, s.cumulative, s.note or ""] for s in plan.splits]
    summary = (
        f"{plan.strategy_summary}\n\n**Predicted finish:** {plan.predicted_finish}"
        f"  ·  **Fade:** {plan.fade_allowance_pct:.1f}%  ·  **Heat adj:** {plan.heat_adjustment_s_per_km:.1f}s/km{cached_note}\n\n"
        + ("**Cautions:** " + "; ".join(plan.cautions) if plan.cautions else "")
        + f"\n\n**Citations:** {', '.join(plan.citations)}"
    )

    voice = audio.voice_for("en") or "en-US-AriaNeural"
    audio_path = audio.speak_cached(plan.strategy_summary, voice, audio_cache)

    dev_panel = packed.report_markdown() + "\n\n**Cache stats:** " + str(semantic_cache.stats())

    return summary, splits_rows, str(audio_path), dev_panel


def _parse_goal_minutes(text: str) -> int | None:
    import re
    text = text.strip().lower()
    m = re.match(r"(\d+):(\d{2})(?::(\d{2}))?", text)
    if m:
        h, mm = int(m.group(1)), int(m.group(2))
        return h * 60 + mm
    m = re.search(r"(\d+)\s*min", text)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*h", text)
    if m:
        return round(float(m.group(1)) * 60)
    return None


with gr.Blocks(title="Race Day Copilot") as demo:
    gr.Markdown("# 🏃 Race Day Copilot\nPhotograph a course elevation profile, tell me your goal, get a verified pacing plan.")

    course_state = gr.State(None)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. Course")
            example_dropdown = gr.Dropdown(
                choices=["flat_berlin_like", "rolling", "hilly_boston_like",
                         "net_downhill", "out_and_back_climb", "half_marathon_flat"],
                label="Try an example course",
            )
            course_image = gr.Image(type="pil", label="Or upload a course elevation profile photo")
            extract_btn = gr.Button("Extract course from photo")
            course_status = gr.Markdown()
            segments_table = gr.Dataframe(
                headers=["km_start", "km_end", "trend", "severity", "note"],
                label="Segments (edit if the extraction looks wrong)",
            )
            distance_input = gr.Number(label="Distance (km)", value=42.2)

            gr.Markdown("### 2. Your goal")
            goal_audio = gr.Audio(sources=["microphone"], type="filepath", label="Speak your goal")
            goal_text = gr.Textbox(label="Goal time (e.g. '3:45' or type after speaking)")
            temp_input = gr.Number(label="Expected temp (°C)", value=None)
            experience_input = gr.Dropdown(choices=["first", "1-3", "4+"], label="Experience", value="1-3")
            units_input = gr.Radio(choices=["km", "mi"], value="km", label="Units")

            build_btn = gr.Button("Build my plan", variant="primary")

        with gr.Column():
            gr.Markdown("### Your plan")
            plan_summary = gr.Markdown()
            splits_output = gr.Dataframe(headers=["km", "target_pace", "cumulative", "note"])
            plan_audio = gr.Audio(label="Listen to your plan")
            with gr.Accordion("Dev panel (context budget + cache stats)", open=False):
                dev_panel_output = gr.Markdown()

    example_dropdown.change(load_example, inputs=[example_dropdown],
                             outputs=[course_image, course_state, course_status, segments_table])
    extract_btn.click(extract_course, inputs=[course_image], outputs=[course_state, course_status, segments_table])
    goal_audio.change(transcribe_goal, inputs=[goal_audio], outputs=[goal_text])
    build_btn.click(
        build_plan,
        inputs=[course_state, segments_table, distance_input, goal_text, temp_input, experience_input, units_input],
        outputs=[plan_summary, splits_output, plan_audio, dev_panel_output],
    )

if __name__ == "__main__":
    demo.launch()
