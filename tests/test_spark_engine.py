import importlib.util

import pytest


pytestmark = pytest.mark.skipif(importlib.util.find_spec("pyspark") is None, reason="PySpark is an optional dependency")


def test_spark_profile_csv(tmp_path):
    from synthetic_data_platform.spark_engine import profile_csv

    source = tmp_path / "events.csv"
    source.write_text("value,region\n1,North\n2,South\n2,South\n", encoding="utf-8")
    report = profile_csv(source)
    assert report["engine"] == "pyspark"
    assert report["rows"] == 3
    assert report["duplicate_rows"] == 1
    assert report["schema"]["value"] in {"int", "bigint"}

