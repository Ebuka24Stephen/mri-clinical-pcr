"""Clinical tabular preprocessing.

All transformers are ``fit`` on the training split only and then ``transform``
the validation/test splits, preventing any data leakage. Non-numeric columns
are automatically one-hot encoded; numeric columns are imputed and standardised.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.logging_setup import get_logger

logger = get_logger(__name__)


class TabularPreprocessor:
    """Fits imputation, standardisation and categorical encoding."""

    def __init__(
        self,
        impute_strategy: str = "median",
        standardize: bool = True,
        categorical_columns: list[str] | None = None,
    ) -> None:
        """Initialise the preprocessor.

        Args:
            impute_strategy: Scikit-learn imputation strategy.
            standardize: Whether to z-score standardise numeric columns.
            categorical_columns: Columns forced to one-hot encoding.
        """
        self.impute_strategy = impute_strategy
        self.standardize = standardize
        self.categorical_columns = list(categorical_columns or [])

        self._numeric_imputer: SimpleImputer | None = None
        self._numeric_scaler: StandardScaler | None = None
        self._cat_encoder: OneHotEncoder | None = None
        self._numeric_columns: list[str] = []
        self._categorical_columns: list[str] = []
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------ #
    @property
    def feature_names(self) -> list[str]:
        """Names of the produced feature columns (for SHAP and reporting)."""
        return list(self._feature_names)

    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        """Fit preprocessing on the training split.

        Args:
            df: Clinical feature data (patients x features).

        Returns:
            ``self`` for chaining.
        """
        df = df.copy()
        self._categorical_columns = [c for c in self.categorical_columns if c in df.columns]
        for col in df.columns:
            if col not in self._categorical_columns and not pd.api.types.is_numeric_dtype(df[col]):
                self._categorical_columns.append(col)
        self._numeric_columns = [c for c in df.columns if c not in self._categorical_columns]

        numeric = df[self._numeric_columns]
        self._numeric_imputer = SimpleImputer(strategy=self.impute_strategy)
        numeric_imputed = self._numeric_imputer.fit_transform(numeric)

        if self._numeric_columns:
            if self.standardize:
                self._numeric_scaler = StandardScaler()
                numeric_imputed = self._numeric_scaler.fit_transform(numeric_imputed)
            numeric_imputed = np.asarray(numeric_imputed, dtype=np.float64)

        if self._categorical_columns:
            categorical = df[self._categorical_columns].astype(str)
            self._cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            cat_encoded = self._cat_encoder.fit_transform(categorical)
            cat_encoded = np.asarray(cat_encoded, dtype=np.float64)
        else:
            cat_encoded = np.empty((len(df), 0), dtype=np.float64)

        self._feature_names = list(self._numeric_columns)
        if self._cat_encoder is not None:
            self._feature_names += [f"{c}_{v}" for c, vals in zip(
                self._categorical_columns, self._cat_encoder.categories_
            ) for v in vals]
        logger.info(
            "TabularPreprocessor fit: %d numeric, %d categorical -> %d features",
            len(self._numeric_columns),
            len(self._categorical_columns),
            len(self._feature_names),
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using the fitted parameters.

        Args:
            df: Data frame with the same columns seen during fit.

        Returns:
            Float64 array of shape ``(n, n_features)``.
        """
        if self._numeric_imputer is None:
            raise RuntimeError("TabularPreprocessor.transform called before fit().")
        numeric = np.asarray(
            self._numeric_imputer.transform(df[self._numeric_columns]), dtype=np.float64
        )
        if self._numeric_scaler is not None:
            numeric = self._numeric_scaler.transform(numeric)

        if self._cat_encoder is not None:
            categorical = df[self._categorical_columns].astype(str)
            cat_encoded = np.asarray(
                self._cat_encoder.transform(categorical), dtype=np.float64
            )
        else:
            cat_encoded = np.empty((len(df), 0), dtype=np.float64)

        return np.hstack([numeric, cat_encoded])

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one call."""
        return self.fit(df).transform(df)