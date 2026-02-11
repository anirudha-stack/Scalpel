"""Generate README + social media visuals for Segmenta.

This script produces:
- docs/images/segmenta_pipeline_overview.png
- docs/images/segmenta_chunking_demo.gif
- docs/media/segmenta_chunking_demo.mp4 (optional, if OpenCV is available)
"""

from __future__ import annotations

import math
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
IMAGES_DIR = DOCS_DIR / "images"
MEDIA_DIR = DOCS_DIR / "media"

WINDOWS_FONTS_DIR = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"


def _load_font(font_filename: str, size: int) -> ImageFont.FreeTypeFont:
    path = WINDOWS_FONTS_DIR / font_filename
    return ImageFont.truetype(str(path), size=size)


def _try_load_font(filenames: Sequence[str], size: int) -> ImageFont.FreeTypeFont:
    for filename in filenames:
        try:
            return _load_font(filename, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> float:
    # textlength gives more consistent results than bbox width on some Pillow versions.
    return float(draw.textlength(text, font=font))


def _text_height(font: ImageFont.ImageFont) -> int:
    bbox = font.getbbox("Ag")
    return int(bbox[3] - bbox[1])


def _wrap_text(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int
) -> List[str]:
    words = text.split()
    lines: List[str] = []
    current: List[str] = []

    for word in words:
        tentative = " ".join([*current, word]).strip()
        if not tentative:
            continue
        if _text_width(draw, tentative, font) <= max_width:
            current.append(word)
            continue

        if current:
            lines.append(" ".join(current))
            current = [word]
        else:
            # Single word longer than max width; hard-wrap.
            lines.extend(textwrap.wrap(word, width=max(1, int(max_width / 10))))

    if current:
        lines.append(" ".join(current))

    return lines


def _rounded_rect(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int, int, int],
    radius: int,
    *,
    fill: str,
    outline: str,
    width: int = 2,
) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def _arrow(
    draw: ImageDraw.ImageDraw,
    start: Tuple[int, int],
    end: Tuple[int, int],
    *,
    color: str,
    width: int = 4,
    head_len: int = 14,
    head_angle_deg: float = 28.0,
) -> None:
    draw.line([start, end], fill=color, width=width)

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.atan2(dy, dx)
    a1 = angle + math.radians(180.0 - head_angle_deg)
    a2 = angle - math.radians(180.0 - head_angle_deg)
    p1 = (int(end[0] + head_len * math.cos(a1)), int(end[1] + head_len * math.sin(a1)))
    p2 = (int(end[0] + head_len * math.cos(a2)), int(end[1] + head_len * math.sin(a2)))
    draw.polygon([end, p1, p2], fill=color)


def generate_pipeline_overview(output_path: Path) -> None:
    width, height = 1800, 700
    bg = "#FFFFFF"

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    title_font = _try_load_font(["segoeuib.ttf", "arialbd.ttf"], size=54)
    subtitle_font = _try_load_font(["segoeui.ttf", "arial.ttf"], size=28)
    box_font = _try_load_font(["segoeui.ttf", "arial.ttf"], size=26)
    small_font = _try_load_font(["segoeui.ttf", "arial.ttf"], size=22)

    title = "Segmenta pipeline (semantic chunking for retrieval)"
    subtitle = "Local parsing + embeddings, with optional LLM planning, validation, and metadata enrichment"

    draw.text((80, 40), title, fill="#0F172A", font=title_font)
    draw.text((80, 105), subtitle, fill="#334155", font=subtitle_font)

    margin_x = 80
    spacing = 40
    box_w = 240
    box_h = 120
    top_y = 190

    steps = [
        ("Input", "PDF / MD / TXT", "local"),
        ("Parse & Segment", "Paragraphs", "local"),
        ("Atomize", "Sentence groups\n(optional)", "local"),
        ("Embeddings", "Adjacent similarity\n(propose splits)", "local"),
        ("Form Chunks", "Apply validated\nboundaries", "local"),
        ("Output", "Markdown chunks\n+ JSONL traces", "output"),
    ]

    local_fill = "#F8FAFC"
    local_outline = "#CBD5E1"
    llm_fill = "#FDF4FF"
    llm_outline = "#A855F7"
    output_fill = "#ECFDF5"
    output_outline = "#10B981"
    arrow_color = "#64748B"

    boxes: List[Tuple[int, int, int, int]] = []
    for i, (header, body, kind) in enumerate(steps):
        x1 = margin_x + i * (box_w + spacing)
        y1 = top_y
        x2 = x1 + box_w
        y2 = y1 + box_h
        boxes.append((x1, y1, x2, y2))

        if kind == "output":
            fill, outline = output_fill, output_outline
        else:
            fill, outline = local_fill, local_outline

        _rounded_rect(draw, (x1, y1, x2, y2), radius=20, fill=fill, outline=outline, width=3)

        # Header
        header_lines = _wrap_text(draw, header, box_font, max_width=box_w - 26)
        body_lines: List[str] = []
        for line in body.splitlines():
            body_lines.extend(_wrap_text(draw, line, small_font, max_width=box_w - 26))

        header_h = _text_height(box_font)
        body_h = _text_height(small_font)
        total_text_h = len(header_lines) * header_h + 10 + len(body_lines) * body_h
        text_y = y1 + (box_h - total_text_h) // 2

        for hl in header_lines:
            w = _text_width(draw, hl, box_font)
            draw.text((x1 + (box_w - w) / 2, text_y), hl, fill="#0F172A", font=box_font)
            text_y += header_h
        text_y += 10
        for bl in body_lines:
            w = _text_width(draw, bl, small_font)
            draw.text((x1 + (box_w - w) / 2, text_y), bl, fill="#475569", font=small_font)
            text_y += body_h

        # Arrow to next box
        if i < len(steps) - 1:
            start = (x2 + 10, y1 + box_h // 2)
            end = (x2 + spacing - 10, y1 + box_h // 2)
            _arrow(draw, start, end, color=arrow_color, width=5)

    # LLM callouts (below)
    callouts = [
        ("LLM: Plan granularity", "Topics, expected chunks,\natomize settings", 2),
        ("LLM: Validate boundaries", "KEEP / MERGE / ADJUST", 3),
        ("LLM: Enrich metadata", "Title, summary, intent,\nkeywords, questions", 4),
    ]
    callout_w, callout_h = 360, 120
    callout_y = 390

    for title, body, target_idx in callouts:
        target_box = boxes[target_idx]
        target_cx = (target_box[0] + target_box[2]) // 2
        x1 = int(target_cx - callout_w / 2)
        x2 = x1 + callout_w
        y1 = callout_y
        y2 = y1 + callout_h

        _rounded_rect(draw, (x1, y1, x2, y2), radius=18, fill=llm_fill, outline=llm_outline, width=3)

        title_lines = _wrap_text(draw, title, box_font, max_width=callout_w - 28)
        body_lines: List[str] = []
        for line in body.splitlines():
            body_lines.extend(_wrap_text(draw, line, small_font, max_width=callout_w - 28))

        title_h = _text_height(box_font)
        body_h = _text_height(small_font)
        total_text_h = len(title_lines) * title_h + 8 + len(body_lines) * body_h
        text_y = y1 + (callout_h - total_text_h) // 2

        for tl in title_lines:
            w = _text_width(draw, tl, box_font)
            draw.text((x1 + (callout_w - w) / 2, text_y), tl, fill="#581C87", font=box_font)
            text_y += title_h
        text_y += 8
        for bl in body_lines:
            w = _text_width(draw, bl, small_font)
            draw.text((x1 + (callout_w - w) / 2, text_y), bl, fill="#6B21A8", font=small_font)
            text_y += body_h

        # Dotted connector to target box
        start = (target_cx, target_box[3] + 12)
        end = (target_cx, y1 - 12)
        dot_color = "#A855F7"
        dash = 10
        gap = 8
        y = start[1]
        while y < end[1]:
            y2_dash = min(y + dash, end[1])
            draw.line([(start[0], y), (start[0], y2_dash)], fill=dot_color, width=4)
            y = y2_dash + gap
        _arrow(draw, (target_cx, y1 - 16), (target_cx, y1 - 6), color=dot_color, width=4, head_len=10)

    # Legend
    legend_x, legend_y = 80, 610
    _rounded_rect(
        draw,
        (legend_x, legend_y, legend_x + 24, legend_y + 24),
        radius=6,
        fill=llm_fill,
        outline=llm_outline,
        width=3,
    )
    draw.text((legend_x + 34, legend_y - 2), "LLM-involved stage", fill="#6B21A8", font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG", optimize=True)


@dataclass(frozen=True)
class ChunkGroup:
    title: str
    sentences: Sequence[int]  # indices into the sentences list
    color_fill: str
    color_outline: str


def _draw_sentence_block(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    max_width: int,
    font: ImageFont.ImageFont,
    index: int,
    sentence: str,
    highlight: Optional[Tuple[str, str]] = None,
) -> Tuple[int, int]:
    pad_x = 16
    pad_y = 12
    line_gap = 6
    num_prefix = f"{index + 1}."

    num_w = int(_text_width(draw, num_prefix, font))
    text_x = x + num_w + 14
    lines = _wrap_text(draw, sentence, font, max_width=max_width - (num_w + 14))
    line_h = _text_height(font)

    block_h = pad_y * 2 + len(lines) * line_h + max(0, len(lines) - 1) * line_gap
    block_w = max_width

    if highlight is not None:
        fill, outline = highlight
        _rounded_rect(
            draw,
            (x, y, x + block_w, y + block_h),
            radius=18,
            fill=fill,
            outline=outline,
            width=3,
        )
    else:
        _rounded_rect(
            draw,
            (x, y, x + block_w, y + block_h),
            radius=18,
            fill="#FFFFFF",
            outline="#E2E8F0",
            width=2,
        )

    draw.text((x + pad_x, y + pad_y), num_prefix, fill="#0F172A", font=font)
    current_y = y + pad_y
    for line in lines:
        draw.text((text_x, current_y), line, fill="#0F172A", font=font)
        current_y += line_h + line_gap

    return x + block_w, y + block_h


def _draw_chunk_card(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    w: int,
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
    group: ChunkGroup,
    chunk_id: int,
    show: bool,
) -> Tuple[int, int]:
    if not show:
        return x + w, y

    pad_x = 18
    pad_y = 14

    header = f"chunk_{chunk_id:03d}"
    title = group.title
    keywords = "keywords: " + ", ".join(
        [kw for kw in title.lower().replace("&", ",").replace("/", ",").split(",") if kw.strip()][:4]
    )

    header_h = _text_height(body_font)
    title_h = _text_height(title_font)
    body_h = _text_height(body_font)

    title_lines = _wrap_text(draw, title, title_font, max_width=w - pad_x * 2)
    keywords_lines = _wrap_text(draw, keywords, body_font, max_width=w - pad_x * 2)

    card_h = pad_y * 2 + header_h + 10 + len(title_lines) * title_h + 8 + len(keywords_lines) * body_h

    _rounded_rect(
        draw,
        (x, y, x + w, y + card_h),
        radius=20,
        fill="#FFFFFF",
        outline=group.color_outline,
        width=4,
    )
    # Accent bar
    _rounded_rect(
        draw,
        (x + 14, y + 14, x + 30, y + card_h - 14),
        radius=8,
        fill=group.color_fill,
        outline=group.color_outline,
        width=2,
    )

    draw.text((x + pad_x + 20, y + pad_y), header, fill="#64748B", font=body_font)
    ty = y + pad_y + header_h + 10
    for line in title_lines:
        draw.text((x + pad_x + 20, ty), line, fill="#0F172A", font=title_font)
        ty += title_h
    ty += 8
    for line in keywords_lines:
        draw.text((x + pad_x + 20, ty), line, fill="#475569", font=body_font)
        ty += body_h

    return x + w, y + card_h


def generate_chunking_demo_gif(output_gif_path: Path, output_mp4_path: Optional[Path] = None) -> None:
    width, height = 1080, 1080
    bg = "#F8FAFC"

    title_font = _try_load_font(["segoeuib.ttf", "arialbd.ttf"], size=54)
    subtitle_font = _try_load_font(["segoeui.ttf", "arial.ttf"], size=28)
    sentence_font = _try_load_font(["segoeui.ttf", "arial.ttf"], size=30)
    card_title_font = _try_load_font(["segoeuib.ttf", "arialbd.ttf"], size=28)
    card_body_font = _try_load_font(["segoeui.ttf", "arial.ttf"], size=22)

    sentences = [
        "Relocation governance defines how mobility policies are approved and enforced across teams.",
        "Eligibility rules clarify who qualifies based on role criticality and tenure.",
        "Budget ownership and reimbursement limits keep relocation costs predictable.",
        "Preferred vendors provide travel and housing services while maintaining quality.",
        "International moves increase compliance complexity (visas, tax treatment, local labor rules).",
        "Tracking outcomes enables continuous improvement of the mobility program.",
    ]

    groups = [
        ChunkGroup(
            title="Governance & eligibility",
            sentences=(0, 1),
            color_fill="#DBEAFE",
            color_outline="#2563EB",
        ),
        ChunkGroup(
            title="Financial planning & vendors",
            sentences=(2, 3),
            color_fill="#DCFCE7",
            color_outline="#16A34A",
        ),
        ChunkGroup(
            title="Compliance & continuous improvement",
            sentences=(4, 5),
            color_fill="#FFEDD5",
            color_outline="#EA580C",
        ),
    ]

    frames: List[Image.Image] = []

    def render_frame(stage: int) -> Image.Image:
        # stage: 0 = input, 1..3 = reveal chunk groups, 4 = final hold
        img = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(img)

        # Header
        draw.text((60, 52), "How Segmenta chunks a long paragraph", fill="#0F172A", font=title_font)
        draw.text(
            (60, 118),
            "Sentence groups → validated boundaries → enriched semantic chunks",
            fill="#475569",
            font=subtitle_font,
        )

        # Columns
        left_x = 60
        col_gap = 50
        col_w = 520
        right_x = left_x + col_w + col_gap
        right_w = width - right_x - 60

        # Column headers
        draw.text((left_x, 178), "Input paragraph (PDF reality)", fill="#334155", font=subtitle_font)
        draw.text((right_x, 178), "Chunks for retrieval", fill="#334155", font=subtitle_font)

        # Sentence blocks
        y = 240
        block_gap = 14
        highlight_by_sentence = {}
        if stage >= 1:
            for gi in range(min(stage, 3)):
                group = groups[gi]
                for idx in group.sentences:
                    highlight_by_sentence[idx] = (group.color_fill, group.color_outline)

        for idx, sentence in enumerate(sentences):
            _draw_sentence_block(
                draw,
                x=left_x,
                y=y,
                max_width=col_w,
                font=sentence_font,
                index=idx,
                sentence=sentence,
                highlight=highlight_by_sentence.get(idx),
            )
            # Estimate next y based on wrapping
            num_prefix = f"{idx + 1}."
            num_w = int(_text_width(draw, num_prefix, sentence_font))
            lines = _wrap_text(draw, sentence, sentence_font, max_width=col_w - (num_w + 14))
            block_h = 12 * 2 + len(lines) * _text_height(sentence_font) + max(0, len(lines) - 1) * 6
            y += block_h + block_gap

        # Chunk cards (reveal progressively)
        card_y = 240
        card_gap = 18
        for gi, group in enumerate(groups):
            show = stage >= (gi + 1)
            _, card_y2 = _draw_chunk_card(
                draw,
                x=right_x,
                y=card_y,
                w=right_w,
                title_font=card_title_font,
                body_font=card_body_font,
                group=group,
                chunk_id=gi + 1,
                show=show,
            )
            if show:
                card_y = card_y2 + card_gap

        # Footer note
        footer = "Colors show the final chunk grouping (topic-pure, retrieval-friendly)."
        draw.text((60, 1018), footer, fill="#64748B", font=card_body_font)

        return img

    stage_sequence = [0, 1, 2, 3, 3, 3]
    for stage in stage_sequence:
        frames.append(render_frame(stage))

    output_gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=[900, 850, 850, 1100, 1100, 1100],
        loop=0,
        optimize=True,
    )

    if output_mp4_path is not None:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore

            output_mp4_path.parent.mkdir(parents=True, exist_ok=True)
            fps = 10
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_mp4_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError("OpenCV VideoWriter failed to open (mp4v codec missing?)")

            # Repeat frames to match durations.
            durations_ms = [900, 850, 850, 1100, 1100, 1100]
            for frame, ms in zip(frames, durations_ms):
                repeat = max(1, int(round((ms / 1000.0) * fps)))
                bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                for _ in range(repeat):
                    writer.write(bgr)
            writer.release()
        except Exception:
            # MP4 is optional; GIF is always produced.
            pass


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    generate_pipeline_overview(IMAGES_DIR / "segmenta_pipeline_overview.png")
    generate_chunking_demo_gif(
        IMAGES_DIR / "segmenta_chunking_demo.gif",
        output_mp4_path=MEDIA_DIR / "segmenta_chunking_demo.mp4",
    )

    print("Generated:")
    print("-", (IMAGES_DIR / "segmenta_pipeline_overview.png").relative_to(ROOT))
    print("-", (IMAGES_DIR / "segmenta_chunking_demo.gif").relative_to(ROOT))
    print("-", (MEDIA_DIR / "segmenta_chunking_demo.mp4").relative_to(ROOT))


if __name__ == "__main__":
    main()

