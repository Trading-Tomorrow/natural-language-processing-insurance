# Insurance Claim API

FastAPI service for insurance claim fraud analysis using a fine-tuned Qwen3 model and YOLO damage detection.

## Setup

Install dependencies (add to your virtualenv or conda env as needed):

```bash
pip install fastapi uvicorn sqlalchemy pydantic ultralytics mlx-lm
```

Optional: copy the mocked environment config:

```bash
cp api/.env.example api/.env
```

## Run

From the repo root:

```bash
uvicorn api.app.main:app --reload
```

Swagger UI is available at `http://localhost:8000/docs`.

## Endpoints

- `POST /cases` create a case with statements
- `POST /cases/{id}/images` upload one or more images
- `POST /cases/{id}/analyze` run Qwen inference
- `GET /cases/{id}` fetch full case

## Notes

- YOLO weights: `runs/detect/runs/detect/models/modelo_large_v3/weights/best.pt`
- Qwen uses MLX subprocess via `mlx_lm generate`
- Images are stored under `data/uploads/`
