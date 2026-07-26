from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class NumericSpec:
    column: str
    mean: float
    std: float
    minimum: float
    maximum: float


@dataclass
class CategoricalSpec:
    column: str
    categories: list[str]


class TabularPreprocessor:
    """Convert mixed tabular data into a neural-network-friendly representation."""

    def __init__(self) -> None:
        self.numeric: list[NumericSpec] = []
        self.categorical: list[CategoricalSpec] = []
        self.feature_slices: dict[str, tuple[int, int]] = {}
        self.column_order: list[str] = []
        self.fitted = False

    def fit(self, frame: pd.DataFrame) -> "TabularPreprocessor":
        self.numeric = []
        self.categorical = []
        self.column_order = [str(column) for column in frame.columns]
        for column in frame.columns:
            series = frame[column]
            if pd.api.types.is_numeric_dtype(series):
                values = pd.to_numeric(series, errors="coerce")
                mean = float(values.mean())
                std = float(values.std(ddof=0))
                self.numeric.append(
                    NumericSpec(
                        column=str(column),
                        mean=mean,
                        std=std if std > 1e-8 else 1.0,
                        minimum=float(values.min()),
                        maximum=float(values.max()),
                    )
                )
            else:
                values = series.fillna("__MISSING__").astype(str)
                categories = sorted(values.unique().tolist())
                self.categorical.append(CategoricalSpec(str(column), categories))

        offset = 0
        self.feature_slices = {}
        for spec in self.numeric:
            self.feature_slices[spec.column] = (offset, offset + 1)
            offset += 1
        for spec in self.categorical:
            self.feature_slices[spec.column] = (offset, offset + len(spec.categories))
            offset += len(spec.categories)
        self.fitted = True
        return self

    @property
    def dimension(self) -> int:
        return sum(end - start for start, end in self.feature_slices.values())

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        output = np.zeros((len(frame), self.dimension), dtype=np.float32)
        for spec in self.numeric:
            values = pd.to_numeric(frame[spec.column], errors="coerce").fillna(spec.mean)
            scaled = (values.to_numpy(dtype=np.float32) - spec.mean) / spec.std
            start, end = self.feature_slices[spec.column]
            output[:, start:end] = np.clip(scaled / 3.0, -1.0, 1.0).reshape(-1, 1)
        for spec in self.categorical:
            start, end = self.feature_slices[spec.column]
            lookup = {category: index for index, category in enumerate(spec.categories)}
            values = frame[spec.column].fillna("__MISSING__").astype(str)
            for row, value in enumerate(values):
                output[row, start + lookup.get(value, 0)] = 1.0
        return output

    def inverse_transform(self, matrix: np.ndarray) -> pd.DataFrame:
        self._check_fitted()
        matrix = np.asarray(matrix)
        if matrix.ndim != 2 or matrix.shape[1] != self.dimension:
            raise ValueError(f"Expected matrix with shape (n, {self.dimension}).")
        result: dict[str, np.ndarray] = {}
        for spec in self.numeric:
            start, end = self.feature_slices[spec.column]
            scaled = np.clip(matrix[:, start], -1.0, 1.0) * 3.0
            values = scaled * spec.std + spec.mean
            result[spec.column] = np.clip(values, spec.minimum, spec.maximum)
        for spec in self.categorical:
            start, end = self.feature_slices[spec.column]
            indexes = np.argmax(matrix[:, start:end], axis=1)
            result[spec.column] = np.array([spec.categories[int(index)] for index in indexes], dtype=object)
        return pd.DataFrame({column: result[column] for column in self.column_order})

    def metadata(self) -> dict:
        return {
            "numeric_columns": [spec.column for spec in self.numeric],
            "categorical_columns": [spec.column for spec in self.categorical],
            "column_order": self.column_order,
            "dimension": self.dimension,
            "categories": {spec.column: spec.categories for spec in self.categorical},
        }

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("TabularPreprocessor.fit must be called first.")
