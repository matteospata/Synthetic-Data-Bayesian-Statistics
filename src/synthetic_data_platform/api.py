from __future__ import annotations

from pathlib import Path

from synthetic_data_platform.pipeline import PipelineConfig, SyntheticDataPipeline

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError:  # Optional dependency: install with pip install -e '.[api]'
    FastAPI = None
    HTTPException = None
    BaseModel = object


class GenerationRequest(BaseModel):
    input_path: str
    output_dir: str = "artifacts"
    model: str = "vae"
    epochs: int = 120
    samples: int = 500


def create_app():
    if FastAPI is None:
        raise RuntimeError("FastAPI is optional. Install the API extras with: pip install -e '.[api]'")
    app = FastAPI(title="Synthetic Data & Bayesian Analytics API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/generate")
    def generate(request: GenerationRequest) -> dict:
        if not Path(request.input_path).exists():
            raise HTTPException(status_code=404, detail="Input CSV not found")
        config = PipelineConfig(model=request.model, epochs=request.epochs, samples=request.samples)
        return SyntheticDataPipeline(config).run(request.input_path, request.output_dir)

    return app


app = create_app() if FastAPI is not None else None

