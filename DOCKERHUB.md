# Fashion Recommendation

Upload a photo of a single garment; CLIP recognizes its type, color, style,
gender and age, and the app assembles complete outfits by fetching matching
product photos from the web (DuckDuckGo image search). No dataset and no
training - zero-shot recognition plus web search.

## Run

```bash
docker run --rm -p 7860:7860 flugmaschine/fashion-recommendation:latest
```

Then open http://127.0.0.1:7860 (requires network access).

## Configuration

- `VFR_HOST` - bind address (default `0.0.0.0` in the image)
- `VFR_PORT` - port (default `7860`)
- `VFR_MODEL` - CLIP model name (default `openai/clip-vit-base-patch32`)

Tags: `latest`, `1.3.0`. Stack: Python 3.13, PyTorch (CPU), Transformers (CLIP),
Gradio. CLIP weights are baked into the image.