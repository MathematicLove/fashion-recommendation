"""Fashion Recommendation web UI: recognize a garment and build outfits from the web."""
from __future__ import annotations
import os
import src  # sets OpenMP flag before torch is imported
import gradio as gr
from PIL import Image
from src.analyze import Analysis, analyze_image, analyze_text, is_clothing
from src.model import get_encoder
from src.recommend import recommend_outfit

# One outfit is shown at a time; up to this many are kept for Previous / Next.
MAX_OUTFITS = 5
MAX_PIECES = 5

NO_CLOTHES = ("Can't find any clothes on photo... "
              "You can describe item or paste another photo")
TEXT_PLACEHOLDER = "black leather jacket"
TEXT_LOCKED = "Not available while a photo is uploaded"
IMAGE_LABEL = "Item photo"
IMAGE_LOCKED = "Item photo (clear the description first)"

INK = "#15171c"
INK_SOFT = "#5c6270"
LINE = "#e2e4e9"
SURFACE = "#ffffff"
PAGE = "#ffffff"
RAISED = "#fafafb"

INK_D = "#e9ebef"
INK_SOFT_D = "#9ba1ac"
LINE_D = "#2b2f38"
SURFACE_D = "#16181d"
PAGE_D = "#0f1115"
RAISED_D = "#1b1e24"

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.gray,
    secondary_hue=gr.themes.colors.gray,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.LocalFont("system-ui"), gr.themes.GoogleFont("Inter"),
          gr.themes.LocalFont("Arial"), gr.themes.LocalFont("sans-serif")],
    font_mono=[gr.themes.LocalFont("ui-monospace"), gr.themes.LocalFont("Consolas"),
               gr.themes.LocalFont("monospace")],
).set(
    body_background_fill=PAGE, body_background_fill_dark=PAGE_D,
    body_text_color=INK, body_text_color_dark=INK_D,
    body_text_color_subdued=INK_SOFT, body_text_color_subdued_dark=INK_SOFT_D,
    background_fill_primary=SURFACE, background_fill_primary_dark=SURFACE_D,
    background_fill_secondary=RAISED, background_fill_secondary_dark=RAISED_D,
    block_background_fill=SURFACE, block_background_fill_dark=SURFACE_D,
    block_border_color=LINE, block_border_color_dark=LINE_D,
    block_border_width="1px",
    block_label_background_fill=SURFACE, block_label_background_fill_dark=SURFACE_D,
    block_label_text_color=INK, block_label_text_color_dark=INK_D,
    block_label_text_weight="500",
    block_title_text_color=INK, block_title_text_color_dark=INK_D,
    block_info_text_color=INK_SOFT, block_info_text_color_dark=INK_SOFT_D,
    block_padding="8px",
    block_radius="8px",
    block_shadow="none",
    border_color_primary=LINE, border_color_primary_dark=LINE_D,
    border_color_accent="#c9ced8", border_color_accent_dark="#3a3f4b",
    panel_background_fill=SURFACE, panel_background_fill_dark=SURFACE_D,
    panel_border_color=LINE, panel_border_color_dark=LINE_D,
    input_background_fill=SURFACE, input_background_fill_dark=RAISED_D,
    input_border_color="#d5d9e0", input_border_color_dark="#333944",
    input_border_width="1px",
    input_placeholder_color="#8b909c", input_placeholder_color_dark="#767d89",
    input_radius="8px",
    input_shadow="none",
    button_border_width="1px",
    button_large_radius="8px", button_small_radius="8px",
    # The primary button inverts with the scheme: dark on light, light on dark.
    button_primary_background_fill=INK, button_primary_background_fill_dark=INK_D,
    button_primary_background_fill_hover="#2f3440",
    button_primary_background_fill_hover_dark="#ffffff",
    button_primary_text_color=SURFACE, button_primary_text_color_dark=INK,
    button_primary_border_color=INK, button_primary_border_color_dark=INK_D,
    button_primary_shadow="none", button_primary_shadow_hover="none",
    button_secondary_background_fill=SURFACE, button_secondary_background_fill_dark=RAISED_D,
    button_secondary_background_fill_hover="#f2f3f5",
    button_secondary_background_fill_hover_dark="#242832",
    button_secondary_text_color=INK, button_secondary_text_color_dark=INK_D,
    button_secondary_text_color_hover=INK, button_secondary_text_color_hover_dark=INK_D,
    button_secondary_border_color="#d5d9e0", button_secondary_border_color_dark="#333944",
    button_secondary_shadow="none", button_secondary_shadow_hover="none",
)

# Colors come from the theme variables, so this stylesheet follows the active scheme.
CSS = """
.gradio-container {max-width: 1040px !important; margin: 0 auto !important;}
#intro h1 {font-size: 1.55rem; font-weight: 600; letter-spacing: -.01em; margin: 0 0 .2rem;}
#intro p {color: var(--body-text-color-subdued); margin: 0; font-size: .93rem;
    line-height: 1.5;}
#status p {color: var(--body-text-color); margin: .1rem 0; font-size: .95rem;
    line-height: 1.5;}
#status strong {font-weight: 600;}
#counter p {color: var(--body-text-color-subdued); margin: .5rem 0 .2rem; font-size: .85rem;}
#pieces {gap: 10px;}
.piece {border: 1px solid var(--border-color-primary) !important; border-radius: 8px !important;
    background: var(--block-background-fill) !important;}
.piece img {object-fit: contain !important;}
.piece .label-wrap span, .piece label span {color: var(--body-text-color) !important;
    font-size: .8rem !important;}
#nav {gap: 8px; margin-top: 14px;}
#nav button {flex: 0 0 auto;}
footer {display: none !important;}
"""


def _recognized(analysis: Analysis | None) -> str:
    if analysis is None:
        return ""
    return f"**Recognized item.** {analysis.describe()}"


def _display_row(row: list, item: Image.Image | None) -> list:
    """Replace the placeholder entry with the uploaded photo, drop it for text queries."""
    out = []
    for img, caption in row:
        img = item if img is None else img
        if img is not None:
            out.append((img, caption))
    return out


def _pieces(row: list) -> list:
    updates = []
    for i in range(MAX_PIECES):
        if i < len(row):
            img, caption = row[i]
            updates.append(gr.update(visible=True, value=img, label=caption))
        else:
            updates.append(gr.update(visible=False, value=None))
    return updates


def _nav(outfits: list, idx: int) -> list:
    """Previous appears once you have moved forward; Next disappears at MAX_OUTFITS."""
    return [
        gr.update(visible=idx > 0),
        gr.update(visible=bool(outfits)
                  and (idx + 1 < len(outfits) or len(outfits) < MAX_OUTFITS)),
        gr.update(visible=len(outfits) >= MAX_OUTFITS),
    ]


def _counter(outfits: list, idx: int) -> str:
    if not outfits:
        return ""
    return f"Outfit {idx + 1} of {len(outfits)}"


def _view(status: str, outfits: list, idx: int, analysis, offset: int, item) -> list:
    row = outfits[idx] if outfits else []
    return [status, _counter(outfits, idx), *_pieces(row), *_nav(outfits, idx),
            outfits, idx, analysis, offset, item]


def _busy(status: str, outfits: list, idx: int, analysis, offset: int, item) -> list:
    """Keep the current pieces on screen and lock navigation while work runs."""
    return [status, _counter(outfits, idx), *[gr.skip()] * MAX_PIECES,
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            outfits, idx, analysis, offset, item]


def run(image: Image.Image | None, text_query: str):
    text_query = (text_query or "").strip()
    if image is None and not text_query:
        raise gr.Error("Upload a photo of an item or enter a text query.")
    yield _busy("Analyzing the item.", [], 0, None, 0, image)
    if image is not None:
        img_vec = get_encoder().encode_images([image])[0]
        if not is_clothing(img_vec):
            yield _view(NO_CLOTHES, [], 0, None, 0, None)
            return
        analysis = analyze_image(image, img_vec)
    else:
        analysis = analyze_text(text_query)
    yield _busy(f"{_recognized(analysis)}\n\nSearching the web for matching items.",
                [], 0, analysis, 0, image)
    try:
        row = _display_row(recommend_outfit(analysis, 0), image)
    except RuntimeError as e:
        gr.Warning(str(e))
        yield _view(_recognized(analysis), [], 0, analysis, 1, image)
        return
    yield _view(_recognized(analysis), [row], 0, analysis, 1, image)


def next_outfit(outfits: list, idx: int, analysis, offset: int, item):
    if analysis is None:
        raise gr.Error("Build an outfit first.")
    if idx + 1 < len(outfits) or len(outfits) >= MAX_OUTFITS:
        yield _view(_recognized(analysis), outfits, min(idx + 1, len(outfits) - 1),
                    analysis, offset, item)
        return
    yield _busy(f"{_recognized(analysis)}\n\nLooking for another outfit.",
                outfits, idx, analysis, offset, item)
    try:
        row = _display_row(recommend_outfit(analysis, offset), item)
    except RuntimeError as e:
        gr.Warning(str(e))
        yield _view(_recognized(analysis), outfits, idx, analysis, offset + 1, item)
        return
    outfits = outfits + [row]
    yield _view(_recognized(analysis), outfits, len(outfits) - 1, analysis, offset + 1, item)


def prev_outfit(outfits: list, idx: int, analysis, offset: int, item):
    return _view(_recognized(analysis), outfits, max(idx - 1, 0), analysis, offset, item)


def fresh_outfits(outfits: list, idx: int, analysis, offset: int, item):
    """Discard the stored outfits and start a new set from unused variants."""
    if analysis is None:
        raise gr.Error("Build an outfit first.")
    yield _busy(f"{_recognized(analysis)}\n\nBuilding a new set of outfits.",
                [], 0, analysis, offset, item)
    try:
        row = _display_row(recommend_outfit(analysis, offset), item)
    except RuntimeError as e:
        gr.Warning(str(e))
        yield _view(_recognized(analysis), outfits, idx, analysis, offset + 1, item)
        return
    yield _view(_recognized(analysis), [row], 0, analysis, offset + 1, item)


def lock_text(image: Image.Image | None):
    """Only one input at a time: a photo disables the description field."""
    if image is None:
        return gr.update(interactive=True, placeholder=TEXT_PLACEHOLDER)
    return gr.update(interactive=False, placeholder=TEXT_LOCKED)


def lock_image(text_query: str):
    """Only one input at a time: a description disables the photo field."""
    if (text_query or "").strip():
        return gr.update(interactive=False, label=IMAGE_LOCKED)
    return gr.update(interactive=True, label=IMAGE_LABEL)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Fashion Recommendation") as demo:
        outfits_state = gr.State([])
        idx_state = gr.State(0)
        analysis_state = gr.State(None)
        offset_state = gr.State(0)
        item_state = gr.State(None)

        with gr.Column(elem_id="intro"):
            gr.Markdown(
                "# Fashion Recommendation\n"
                "Upload a photo of a single garment or describe it. CLIP recognizes the "
                "item and assembles one complete outfit from matching pieces found on the "
                "web. Use Next to build another one, up to five kept at a time."
            )
        with gr.Row(equal_height=False):
            with gr.Column(scale=2, min_width=260):
                image_in = gr.Image(type="pil", label=IMAGE_LABEL, height=260, buttons=[])
                text_in = gr.Textbox(label="Or describe an item",
                                     placeholder=TEXT_PLACEHOLDER)
                run_btn = gr.Button("Build outfit", variant="primary")
            with gr.Column(scale=3, min_width=380):
                status_md = gr.Markdown(elem_id="status")
                counter_md = gr.Markdown(elem_id="counter")
                with gr.Row(elem_id="pieces", equal_height=True):
                    pieces = [
                        gr.Image(visible=False, interactive=False, height=170,
                                 min_width=120, buttons=["fullscreen"], elem_classes="piece")
                        for _ in range(MAX_PIECES)
                    ]
                with gr.Row(elem_id="nav"):
                    prev_btn = gr.Button("Previous", visible=False, min_width=110)
                    next_btn = gr.Button("Next", visible=False, min_width=110)
                    fresh_btn = gr.Button("I did not like anything", visible=False,
                                          min_width=200)

        states = [outfits_state, idx_state, analysis_state, offset_state, item_state]
        outputs = [status_md, counter_md, *pieces, prev_btn, next_btn, fresh_btn, *states]
        # Progress is shown on the pieces only, so it never overlaps the status text.
        common = dict(show_progress="minimal", show_progress_on=pieces)
        # Each lock listens on the other field's value and writes only its props,
        # so they cannot trigger each other. always_last matters while typing: with
        # the default "once" the keystroke that empties the box is dropped if an
        # earlier call is still running, and the field stays locked.
        image_in.change(lock_text, inputs=image_in, outputs=text_in,
                        show_progress="hidden", trigger_mode="always_last")
        text_in.change(lock_image, inputs=text_in, outputs=image_in,
                       show_progress="hidden", trigger_mode="always_last")
        run_btn.click(run, inputs=[image_in, text_in], outputs=outputs, **common)
        next_btn.click(next_outfit, inputs=states, outputs=outputs, **common)
        prev_btn.click(prev_outfit, inputs=states, outputs=outputs, show_progress="hidden")
        fresh_btn.click(fresh_outfits, inputs=states, outputs=outputs, **common)
    return demo


if __name__ == "__main__":
    server_name = os.environ.get("VFR_HOST", "127.0.0.1")
    server_port = int(os.environ.get("VFR_PORT", "7860"))
    get_encoder()  # load CLIP before serving, so the first request does not wait for it
    build_ui().launch(server_name=server_name, server_port=server_port, theme=THEME, css=CSS)