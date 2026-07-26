import pandas as pd

from synthetic_data_platform.models import TabularGAN, TabularVAE
from synthetic_data_platform.preprocessing import TabularPreprocessor


def _data():
    frame = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6], "group": ["a", "a", "b", "b", "a", "b"]})
    processor = TabularPreprocessor().fit(frame)
    return frame, processor, processor.transform(frame)


def test_vae_generates_expected_shape():
    frame, processor, matrix = _data()
    model = TabularVAE(processor.dimension, latent_dim=2, hidden_dim=16)
    history = model.fit(matrix, processor, epochs=3, batch_size=3, seed=7)
    generated = processor.inverse_transform(model.sample(10, processor, seed=8))
    assert len(history) == 3
    assert generated.shape == (10, len(frame.columns))
    assert set(generated["group"]).issubset({"a", "b"})


def test_gan_generates_expected_shape():
    frame, processor, matrix = _data()
    model = TabularGAN(processor.dimension, latent_dim=2, hidden_dim=16)
    model.fit(matrix, processor, epochs=2, batch_size=3, seed=7)
    generated = processor.inverse_transform(model.sample(5, processor, seed=8))
    assert generated.shape == (5, len(frame.columns))

