"""Fashion Recommendation web UI: recognize a garment and build outfits from the web."""
from __future__ import annotations
import os
import src  # sets OpenMP flag before torch is imported
import gradio as gr
from PIL import Image
from src import config
from src.analyze import analyze_image, analyze_text
from src.model import get_encoder
from src.recommend import recommend_outfits

MAX_OUTFITS = 4
MAX_PIECES = 5

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.slate,
    secondary_hue=gr.themes.colors.gray,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.LocalFont("system-ui"), gr.themes.GoogleFont("Inter"),
          gr.themes.LocalFont("Arial"), gr.themes.LocalFont("sans-serif")],
    font_mono=[gr.themes.LocalFont("ui-monospace"), gr.themes.LocalFont("Consolas"),
               gr.themes.LocalFont("monospace")],
).set(
    body_background_fill="#f6f6f7",
    block_background_fill="#ffffff",
    block_border_width="1px",
    block_shadow="0 1px 2px rgba(16,24,40,.06)",
    block_radius="14px",
    button_primary_background_fill="#1f2430",
    button_primary_background_fill_hover="#333a49",
    button_primary_text_color="#ffffff",
    input_radius="10px",
)

CSS = """
.gradio-container {max-width: 1080px !important; margin: 0 auto !important;}
#intro h1 {font-weight: 650; letter-spacing: -.02em; margin-bottom: .25rem;}
#intro p {color: #667085; margin-top: 0;}
#status {min-height: 1.5rem;}
#status h3 {font-weight: 600; margin: .15rem 0; color: #1f2430;}
.outfit-card {border: 1px solid #e6e8ec !important; border-radius: 14px !important;
    padding: 6px !important; background: #fff !important;}
.outfit-card .label-wrap span {font-weight: 600; color: #1f2430;}
.thumbnail-item img {border-radius: 10px !important;}
footer {display: none !important;}
"""


def _gallery(row: list, input_image: Image.Image | None):
    media = []
    for item, caption in row:
        if item is None:
            if input_image is not None:
                media.append((input_image, caption))
        else:
            media.append((item, caption))
    return media


def _hide_all():
    return [gr.update(visible=False, value=None) for _ in range(MAX_OUTFITS)]


def run(image: Image.Image | None, text_query: str):
    text_query = (text_query or "").strip()
    if image is None and not text_query:
        raise gr.Error("Upload a photo of an item or enter a text query.")
    yield [gr.update(value="### Analyzing the item...")] + _hide_all()
    enc = get_encoder()
    if image is not None:
        analysis = analyze_image(image, enc.encode_images([image])[0])
    else:
        analysis = analyze_text(text_query)
    yield [gr.update(value=(
        f"### Recognized: {analysis.describe()}\n"
        "### Searching the web for matching items..."
    ))] + _hide_all()
    try:
        outfits = recommend_outfits(analysis, n_outfits=MAX_OUTFITS)
    except RuntimeError as e:
        raise gr.Error(str(e))
    rows = [_gallery(o, image) for o in outfits]
    header = (
        f"### Recognized: {analysis.describe()}\n"
        f"### {len(rows)} outfits with this item (items fetched from the web)"
    )
    if not rows:
        header += "\n\n_Could not assemble an outfit._"
    updates = [gr.update(value=header)]
    for i in range(MAX_OUTFITS):
        if i < len(rows):
            updates.append(gr.update(visible=True, label=f"Outfit {i + 1}", value=rows[i]))
        else:
            updates.append(gr.update(visible=False, value=None))
    yield updates


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Fashion Recommendation") as demo:
        with gr.Column(elem_id="intro"):
            gr.Markdown(
                "# Fashion Recommendation\n"
                "Upload a photo of a single garment or type what you have. CLIP "
                "recognizes it and the app assembles complete outfits from matching "
                "items found on the web."
            )
        with gr.Row(equal_height=False):
            with gr.Column(scale=2, min_width=280):
                image_in = gr.Image(type="pil", label="Item photo", height=320)
                text_in = gr.Textbox(label="Or describe an item",
                                     placeholder="e.g. black leather jacket")
                run_btn = gr.Button("Build outfits", variant="primary", size="lg")
            with gr.Column(scale=3, min_width=360):
                header_md = gr.Markdown(elem_id="status")
                galleries = [
                    gr.Gallery(visible=False, columns=MAX_PIECES, height=220,
                               object_fit="contain", show_label=True,
                               elem_classes="outfit-card")
                    for _ in range(MAX_OUTFITS)
                ]
        # Progress is shown only on the galleries, so it never overlaps the status text.
        run_btn.click(run, inputs=[image_in, text_in], outputs=[header_md, *galleries],
                      show_progress="minimal", show_progress_on=galleries)
    return demo


if __name__ == "__main__":
    server_name = os.environ.get("VFR_HOST", "127.0.0.1")
    server_port = int(os.environ.get("VFR_PORT", "7860"))
    build_ui().launch(server_name=server_name, server_port=server_port, theme=THEME, css=CSS)
