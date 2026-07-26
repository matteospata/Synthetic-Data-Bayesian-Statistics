import pandas as pd

from synthetic_data_platform.preprocessing import TabularPreprocessor


def test_preprocessor_round_trip_preserves_schema_and_categories():
    frame = pd.DataFrame({"age": [20, 30, 40], "region": ["North", "South", "North"]})
    preprocessor = TabularPreprocessor().fit(frame)
    reconstructed = preprocessor.inverse_transform(preprocessor.transform(frame))
    assert list(reconstructed.columns) == ["age", "region"]
    assert set(reconstructed["region"]) == {"North", "South"}
    assert reconstructed["age"].between(20, 40).all()

