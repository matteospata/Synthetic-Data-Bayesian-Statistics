import pandas as pd

from synthetic_data_platform.pipeline import PipelineConfig, SyntheticDataPipeline


def test_pipeline_writes_reproducible_artifacts(tmp_path):
    frame = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6], "group": ["a", "a", "b", "b", "a", "b"]})
    result = SyntheticDataPipeline(PipelineConfig(epochs=2, samples=12, latent_dim=2)).run_dataframe(frame, tmp_path)
    artifact_dir = tmp_path / result["run_id"]
    assert (artifact_dir / "synthetic.csv").exists()
    assert (artifact_dir / "quality_report.json").exists()
    assert (artifact_dir / "bayesian_report.json").exists()
    assert result["quality"]["rows"]["synthetic"] == 12

