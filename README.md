# Fashion Recommendation

Upload a photo of a single garment (or type a query). CLIP recognizes its type,
color and style, and the app assembles several complete outfits by fetching
matching items from the web.

## How it works

1. Recognition. The input is encoded with CLIP (`openai/clip-vit-base-patch32`,
   zero-shot, no fine-tuning). For an image, the garment type, color and style
   are inferred by comparing its embedding to text prompts; for text, they are
   parsed by keyword.
2. Assembly. From the recognized role, complementary slots are chosen (for a top:
   bottom, footwear, outerwear, accessory) and matching images are retrieved per
   slot from the Openverse image API (no key, no authentication).
3. Output. Up to four outfits are shown, each as a row of images: the input item
   plus one web image per slot.

There is no local dataset and no training: the model is used only for zero-shot
recognition, and all recommended items come from the web.

## Run with Python

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:7860. Requires network access.

## Run with Docker

```bash
docker run --rm -p 7860:7860 flugmaschine/fashion-recommendation:latest
```

Open http://127.0.0.1:7860. Or use `docker compose up` to build and run locally.

## Configuration

| Variable | Purpose | Default |
| --- | --- | --- |
| `VFR_MODEL` | CLIP model name | `openai/clip-vit-base-patch32` |
| `VFR_HOST` | Server bind address | `127.0.0.1` (`0.0.0.0` in Docker) |
| `VFR_PORT` | Server port | `7860` |

## Structure

```text
src/
  config.py     roles, outfit slots, recognition vocabularies
  model.py      CLIP wrapper (image and text embeddings)
  analyze.py    input recognition: type / color / style / gender
  recommend.py  outfit assembly from web images (Openverse)
app.py          Gradio web UI
Dockerfile      CPU inference image
```
