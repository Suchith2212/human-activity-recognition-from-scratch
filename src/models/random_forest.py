"""
Random Forest Classifier — implemented from scratch using NumPy.

This module provides a bootstrap-aggregating (bagging) ensemble of
:class:`DecisionTree` classifiers.  Each tree is trained on a random
bootstrap sample of the data **and** considers only a random subset of
features at each split (controlled by ``max_features``).  Predictions
are made by majority vote across all trees.

Key Concepts:
    - **Bootstrap Aggregation (Bagging)**: Each tree sees a different
      random sample (drawn *with* replacement), which reduces variance.
    - **Feature Randomisation**: Each split node only evaluates
      ``max_features`` candidate features (default: sqrt(n_features)),
      which *decorrelates* the trees and makes the ensemble stronger.
    - **Majority Voting**: The final prediction is the class that
      receives the most votes across all individual trees.

Why this works:
    A single decision tree has high variance — small changes in the
    training data can produce very different trees.  By averaging
    many decorrelated trees, Random Forest dramatically reduces
    variance while keeping bias low, leading to significantly better
    generalisation performance.

References:
    - Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.
"""

import logging
from typing import Literal, Optional

import numpy as np

from src.models.decision_tree import DecisionTree

logger = logging.getLogger(__name__)


class RandomForest:
    """Random Forest Classifier built from scratch.

    An ensemble of decision trees, each trained on a bootstrap sample
    of the data with random feature subsetting at each split.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int, default=10
        Maximum depth of each individual tree.
    min_samples_split : int, default=2
        Minimum samples required to split an internal node in each tree.
    max_features : int or None, default=None
        Number of features to consider at each split.  If ``None``,
        defaults to ``int(sqrt(n_features))`` (standard RF heuristic).
    criterion : {"entropy", "gini"}, default="entropy"
        Impurity function used for splitting.
    bootstrap : bool, default=True
        Whether to use bootstrap sampling.  If ``False``, each tree
        sees the full training set (only feature randomisation applies).
    random_state : int or None, default=None
        Seed for reproducibility.

    Attributes
    ----------
    trees_ : list[DecisionTree]
        The fitted individual decision trees.
    n_features_ : int
        Number of features in the training data.
    classes_ : np.ndarray
        Unique class labels found during fitting.

    Examples
    --------
    >>> from src.models.random_forest import RandomForest
    >>> rf = RandomForest(n_estimators=50, max_depth=8)
    >>> rf.fit(X_train, y_train)
    >>> predictions = rf.predict(X_test)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 2,
        max_features: Optional[int] = None,
        criterion: Literal["entropy", "gini"] = "entropy",
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.random_state = random_state

        # Fitted attributes
        self.trees_: list[DecisionTree] = []
        self.n_features_: int = 0
        self.classes_: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        return (
            f"RandomForest(n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, "
            f"max_features={self.max_features}, "
            f"criterion='{self.criterion}')"
        )

    # ================================================================
    # Public API
    # ================================================================

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForest":
        """Build the random forest from training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Training labels (integer-encoded).

        Returns
        -------
        self
            The fitted random forest instance.
        """
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=int)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix X, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Lengths of X and y must match. Got X: {X.shape[0]}, y: {y.shape[0]}"
            )

        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.classes_ = np.unique(y)

        # Default max_features = sqrt(n_features)
        max_feat = self.max_features
        if max_feat is None:
            max_feat = max(1, int(np.sqrt(n_features)))

        rng = np.random.RandomState(self.random_state)
        self.trees_ = []

        for i in range(self.n_estimators):
            # Generate a unique seed for each tree from the master RNG
            tree_seed = rng.randint(0, 2**31 - 1)

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                max_features=max_feat,
                random_state=tree_seed,
            )

            # Bootstrap sample (draw n_samples WITH replacement)
            if self.bootstrap:
                indices = rng.choice(n_samples, size=n_samples, replace=True)
                X_boot, y_boot = X[indices], y[indices]
            else:
                X_boot, y_boot = X, y

            tree.fit(X_boot, y_boot)
            self.trees_.append(tree)

            if (i + 1) % 25 == 0 or (i + 1) == self.n_estimators:
                logger.info("  Fitted tree %d / %d", i + 1, self.n_estimators)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels by majority vote across all trees.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix to predict.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        if len(self.trees_) == 0:
            raise RuntimeError("Forest has not been fitted. Call .fit() first.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Collect predictions from each tree: shape (n_estimators, n_samples)
        all_preds = np.array([tree.predict(X) for tree in self.trees_])

        # Majority vote for each sample
        n_samples = X.shape[0]
        final_preds = np.empty(n_samples, dtype=int)
        for i in range(n_samples):
            votes = all_preds[:, i]
            # Use bincount with offset for labels that may not start at 0
            offset = votes.min()
            counts = np.bincount(votes - offset)
            final_preds[i] = int(counts.argmax() + offset)

        return final_preds

    @property
    def feature_importances_(self) -> np.ndarray:
        """Averaged feature importances across all trees.

        Returns
        -------
        np.ndarray of shape (n_features,)
            Mean normalised importance of each feature.
        """
        if len(self.trees_) == 0:
            raise RuntimeError("Forest has not been fitted. Call .fit() first.")

        importances = np.mean(
            [tree.feature_importances_ for tree in self.trees_], axis=0,
        )
        # Re-normalise so they sum to 1.0
        total = importances.sum()
        if total > 0:
            importances /= total
        return importances
