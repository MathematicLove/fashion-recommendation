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
THEME = gr.themes.Default(
    font=["system-ui", "Arial", "sans-serif"],
    font_mono=["ui-monospace", "Consolas", "monospace"],
)

def _gallery(row: list, input_image: Image.Image | None):
    media = []
    for url, caption in row:
        if url is None:
            if input_image is not None:
                media.append((input_image, caption))
        else:
            media.append((url, caption))
    return media

def run(image: Image.Image | None, text_query: str):
    text_query = (text_query or "").strip()
    if image is None and not text_query:
        raise gr.Error("Upload a photo of an item or enter a text query.")
    enc = get_encoder()
    if image is not None:
        analysis = analyze_image(image, enc.encode_images([image])[0])
    else:
        analysis = analyze_text(text_query)
    try:
        outfits = recommend_outfits(analysis, n_outfits=MAX_OUTFITS)
    except RuntimeError as e:
        raise gr.Error(str(e))
    rows = [_gallery(o, image) for o in outfits]
    header = (
        f"### Recognized: **{analysis.describe()}**\n"
        f"### {len(rows)} outfits with this item (items fetched from the web)"
    )
    if not rows:
        header += "\n\n_Could not assemble an outfit._"
    updates = [gr.update(value=header)]
    for i in range(MAX_OUTFITS):
        if i < len(rows):
            updates.append(gr.update(visible=True, label=f"Outfit #{i + 1}", value=rows[i]))
        else:
            updates.append(gr.update(visible=False, value=None))
    return updates

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Fashion Recommendation") as demo:
        gr.Markdown(
            "# Fashion Recommendation - build the outfit\n"
            "Upload a photo of any single garment (t-shirt, jeans, jacket, dress, "
            "shoes...). CLIP recognizes it and the app assembles several complete "
            "outfits from matching items fetched from the web."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="pil", label="Item photo", height=340)
                text_in = gr.Textbox(label="Or a text query", placeholder="e.g. black leather jacket")
                run_btn = gr.Button("Build outfits", variant="primary")
            with gr.Column(scale=2):
                header_md = gr.Markdown()
                galleries = [
                    gr.Gallery(visible=False, columns=MAX_PIECES, height=230,
                               object_fit="contain", show_label=True)
                    for _ in range(MAX_OUTFITS)
                ]
        run_btn.click(run, inputs=[image_in, text_in], outputs=[header_md, *galleries])
    return demo

if __name__ == "__main__":
    server_name = os.environ.get("VFR_HOST", "127.0.0.1")
    server_port = int(os.environ.get("VFR_PORT", "7860"))
    build_ui().launch(theme=THEME, server_name=server_name, server_port=server_port)