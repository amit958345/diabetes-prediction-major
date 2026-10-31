# ==========================================
# IQR OUTLIER CLIPPER
# ==========================================

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class IQRClipper(BaseEstimator, TransformerMixin):

    """
    Custom transformer that performs IQR-based
    outlier clipping.

    The IQR boundaries are learned during fit()
    and reused during transform().
    """

    def __init__(self):

        # Dictionary for lower boundaries.
        self.lower_bounds_ = {}

        # Dictionary for upper boundaries.
        self.upper_bounds_ = {}

    def fit(self, X, y=None):

        # Convert input into DataFrame.
        X = pd.DataFrame(X)

        # Calculate IQR boundaries for each column.
        for column in X.columns:

            # First quartile.
            Q1 = X[column].quantile(0.25)

            # Third quartile.
            Q3 = X[column].quantile(0.75)

            # Interquartile range.
            IQR = Q3 - Q1

            # Calculate lower boundary.
            lower_bound = Q1 - 1.5 * IQR

            # Calculate upper boundary.
            upper_bound = Q3 + 1.5 * IQR

            # Save learned boundaries.
            self.lower_bounds_[column] = lower_bound
            self.upper_bounds_[column] = upper_bound

        return self

    def transform(self, X):

        # Convert input into DataFrame.
        X = pd.DataFrame(X).copy()

        # Apply the learned boundaries.
        for column in X.columns:

            X[column] = X[column].clip(
                self.lower_bounds_[column],
                self.upper_bounds_[column]
            )

        # Return transformed data.
        return X.values