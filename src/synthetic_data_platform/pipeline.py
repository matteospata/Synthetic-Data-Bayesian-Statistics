from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from synthetic_data_platform.bayesian import compare_posteriors
from synthetic_data_platform.config import settings
from synthetic_data_platform.evaluation import compare_tables
from synthetic_data_platform.io import write_json
from synthetic_data_platform.models import TabularGAN, TabularVAE
from synthetic_data_platform.preprocessing import TabularPreprocessor


@dataclass
class PipelineConfig:
    model: str = "vae"
    epochs: int = settings.default_epochs
    samples: int = 500
    seed: int = settings.seed
    device: str = settings.device
    latent_dim: int = 8
    use_spark: bool = False


class SyntheticDataPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    def run(self, input_path: str | Path, output_dir: str | Path) -> dict:
        spark_profile = None
        if self.config.use_spark:
            from synthetic_data_platform.spark_engine import profile_csv

            spark_profile = profile_csv(input_path)
        real = pd.read_csv(input_path)
        result = self.run_dataframe(real, output_dir, source=str(input_path))
        if spark_profile is not None:
            from synthetic_data_platform.io import write_json

            write_json(spark_profile, Path(result["artifact_dir"]) / "spark_profile.json")
        return result

    def run_dataframe(self, real: pd.DataFrame, output_dir: str | Path, source: str = "in-memory") -> dict:
        if self.config.model not in {"vae", "gan"}:
            raise ValueError("model must be either 'vae' or 'gan'.")
        preprocessor = TabularPreprocessor().fit(real)
        matrix = preprocessor.transform(real)
        if self.config.model == "vae":
            model = TabularVAE(preprocessor.dimension, self.config.latent_dim)
            history = model.fit(matrix, preprocessor, epochs=self.config.epochs, seed=self.config.seed, device=self.config.device)
            synthetic_matrix = model.sample(self.config.samples, preprocessor, seed=self.config.seed + 1, device=self.config.device)
        else:
            model = TabularGAN(preprocessor.dimension, self.config.latent_dim)
            history = model.fit(matrix, preprocessor, epochs=self.config.epochs, seed=self.config.seed, device=self.config.device)
            synthetic_matrix = model.sample(self.config.samples, preprocessor, seed=self.config.seed + 1, device=self.config.device)
        synthetic = preprocessor.inverse_transform(synthetic_matrix)
        quality = compare_tables(real, synthetic)
        bayesian = compare_posteriors(real, synthetic, seed=self.config.seed)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = Path(output_dir) / run_id
        destination.mkdir(parents=True, exist_ok=True)
        synthetic.to_csv(destination / "synthetic.csv", index=False)
        write_json(quality, destination / "quality_report.json")
        write_json(bayesian, destination / "bayesian_report.json")
        metadata = {"run_id": run_id, "source": source, "model": self.config.model, "config": self.config.__dict__, "preprocessor": preprocessor.metadata(), "training": {"epochs": self.config.epochs, "final_loss": history[-1] if self.config.model == "vae" else {key: values[-1] for key, values in history.items()}}}
        write_json(metadata, destination / "metadata.json")
        return {"run_id": run_id, "artifact_dir": str(destination), "quality": quality, "bayesian": bayesian}
