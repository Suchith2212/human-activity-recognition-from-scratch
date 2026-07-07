"""
Decision Tree Classifier — implemented from scratch using NumPy.

This module provides a complete, production-quality implementation of the
CART (Classification and Regression Trees) algorithm with support for
both **entropy** (information gain) and **Gini impurity** split criteria.

Key Design Decisions:
    - Pure NumPy implementation — no scikit-learn dependency for the core algo.
    - Supports ``feature_importances_`` property for interpretability.
    - Provides ``get_depth()`` and ``get_n_leaves()`` for tree inspection.
    - Follows the scikit-learn API conventions (``fit`` / ``predict``).

Algorithm Complexity:
    - Training:   O(n_samples × n_features × n_thresholds × depth)
    - Prediction:  O(n_samples × depth)

References:
    - Breiman, L. (1984). Classification and Regression Trees.
    - Quinlan, J.R. (1986). Induction of Decision Trees. Machine Learning.
"""

from typing import Literal, Optional

import numpy as np


class Node:
    """A single node in the decision tree.

    Attributes
    ----------
    feature_index : int or None
        Index of the feature used for splitting (``None`` for leaf nodes).
    threshold : float or None
        Split threshold value (``None`` for leaf nodes).
    left : Node or None
        Left child (samples where ``feature <= threshold``).
    right : Node or None
        Right child (samples where ``feature > threshold``).
    value : int or None
        Predicted class label (only set for leaf nodes).
    n_samples : int
        Number of training samples that reached this node.
    impurity : float
        Impurity of the node (entropy or Gini).
    """

    __slots__ = [
        "feature_index", "threshold", "left", "right",
        "value", "n_samples", "impurity",
    ]

    def __init__(
        self,
        feature_index: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["Node"] = None,
        right: Optional["Node"] = None,
        value: Optional[int] = None,
        n_samples: int = 0,
        impurity: float = 0.0,
    ) -> None:
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.n_samples = n_samples
        self.impurity = impurity

    @property
    def is_leaf(self) -> bool:
        """Check if this node is a leaf (terminal) node."""
        return self.value is not None


class DecisionTree:
    """Decision Tree Classifier built from scratch.

    Implements the CART algorithm with configurable split criterion,
    maximum depth, and minimum samples per split.

    Parameters
    ----------
    max_depth : int, default=5
        Maximum depth of the tree.  Deeper trees can overfit.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    criterion : {"entropy", "gini"}, default="entropy"
        Function to measure the quality of a split.
    max_features : int or None, default=None
        Number of features to consider when looking for the best split.
        If ``None``, all features are considered.  Set to ``int(sqrt(n))``
        for Random Forest decorrelation.
    random_state : int or None, default=None
        Seed for the random number generator used in feature subsetting.

    Attributes
    ----------
    root_ : Node
        The root node of the fitted tree.
    n_classes_ : int
        Number of unique classes found during fitting.
    n_features_ : int
        Number of features in the training data.
    feature_importances_ : np.ndarray
        Normalized total impurity reduction contributed by each feature.

    Examples
    --------
    >>> from src.models.decision_tree import DecisionTree
    >>> tree = DecisionTree(max_depth=3, criterion="gini")
    >>> tree.fit(X_train, y_train)
    >>> predictions = tree.predict(X_test)
    >>> tree.get_depth()
    3
    """

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        criterion: Literal["entropy", "gini"] = "entropy",
        max_features: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        if criterion not in ("entropy", "gini"):
            raise ValueError(
                f"Invalid criterion '{criterion}'. Must be 'entropy' or 'gini'."
            )

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)

        # Fitted attributes (populated after .fit())
        self.root_: Optional[Node] = None
        self.n_classes_: int = 0
        self.n_features_: int = 0
        self._feature_importances: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        return (
            f"DecisionTree(max_depth={self.max_depth}, "
            f"min_samples_split={self.min_samples_split}, "
            f"criterion='{self.criterion}')"
        )

    # ================================================================
    # Public API
    # ================================================================

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        """Build the decision tree from training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Training labels (integer-encoded).

        Returns
        -------
        self
            The fitted decision tree instance.
        """
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=int)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix X, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"Expected 1D label array y, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Lengths of X and y must match. Got X: {X.shape[0]}, y: {y.shape[0]}"
            )
        if len(y) == 0:
            raise ValueError("Training data cannot be empty.")

        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        self._feature_importances = np.zeros(self.n_features_, dtype=np.float64)

        self.root_ = self._build_tree(X, y, depth=0)

        # Normalize feature importances
        total = self._feature_importances.sum()
        if total > 0:
            self._feature_importances /= total

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for the given samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix to predict.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted class labels.

        Raises
        ------
        RuntimeError
            If the tree has not been fitted yet.
        """
        if self.root_ is None:
            raise RuntimeError("Tree has not been fitted. Call .fit() first.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected 2D array with {self.n_features_} features, "
                f"got shape {X.shape}"
            )
        return np.array([self._traverse(x, self.root_) for x in X])

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances based on total impurity reduction.

        Returns
        -------
        np.ndarray of shape (n_features,)
            Normalized importance of each feature.  Sums to 1.0.
        """
        if self._feature_importances is None:
            raise RuntimeError("Tree has not been fitted. Call .fit() first.")
        return self._feature_importances.copy()

    def get_depth(self) -> int:
        """Return the maximum depth of the fitted tree."""
        if self.root_ is None:
            return 0
        return self._compute_depth(self.root_)

    def get_n_leaves(self) -> int:
        """Return the total number of leaf nodes."""
        if self.root_ is None:
            return 0
        return self._count_leaves(self.root_)

    # ================================================================
    # Tree Construction (Private)
    # ================================================================

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """Recursively build the decision tree."""
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        current_impurity = self._impurity(y)

        # --- Stopping conditions ---
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            return Node(
                value=self._most_common_label(y),
                n_samples=n_samples,
                impurity=current_impurity,
            )

        # --- Find the best split ---
        best_feat, best_thresh, best_gain = self._best_split(X, y, n_features)

        if best_feat is None or best_gain <= 0:
            return Node(
                value=self._most_common_label(y),
                n_samples=n_samples,
                impurity=current_impurity,
            )

        # --- Record feature importance ---
        self._feature_importances[best_feat] += best_gain * n_samples

        # --- Partition and recurse ---
        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(
            feature_index=best_feat,
            threshold=best_thresh,
            left=left_child,
            right=right_child,
            n_samples=n_samples,
            impurity=current_impurity,
        )

    def _best_split(
        self, X: np.ndarray, y: np.ndarray, n_features: int
    ) -> tuple[Optional[int], Optional[float], float]:
        """Find the feature and threshold that yield the highest information gain.

        Uses midpoints between consecutive sorted unique values as candidate
        thresholds, which is the standard CART approach.  This halves the
        search space compared to using raw unique values and avoids ties.

        When ``max_features`` is set, only a random subset of features is
        evaluated at each split, which decorrelates trees in a Random Forest.
        """
        best_gain = -1.0
        best_idx: Optional[int] = None
        best_thresh: Optional[float] = None

        # Select feature indices to evaluate
        if self.max_features is not None and self.max_features < n_features:
            feature_indices = self._rng.choice(
                n_features, size=self.max_features, replace=False,
            )
        else:
            feature_indices = np.arange(n_features)

        for feat_idx in feature_indices:
            sorted_unique = np.unique(X[:, feat_idx])
            # Use midpoints between consecutive values as candidate thresholds
            midpoints = (sorted_unique[:-1] + sorted_unique[1:]) / 2.0
            for thresh in midpoints:
                gain = self._information_gain(y, X[:, feat_idx], thresh)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = feat_idx
                    best_thresh = thresh

        return best_idx, best_thresh, best_gain

    def _information_gain(
        self, y: np.ndarray, feature_col: np.ndarray, threshold: float
    ) -> float:
        """Compute the information gain from splitting on the given threshold."""
        parent_impurity = self._impurity(y)

        left_mask = feature_col <= threshold
        right_mask = ~left_mask

        n_left, n_right = left_mask.sum(), right_mask.sum()
        if n_left == 0 or n_right == 0:
            return 0.0

        n_total = len(y)
        child_impurity = (
            (n_left / n_total) * self._impurity(y[left_mask])
            + (n_right / n_total) * self._impurity(y[right_mask])
        )

        return parent_impurity - child_impurity

    # ================================================================
    # Impurity Measures
    # ================================================================

    def _impurity(self, y: np.ndarray) -> float:
        """Compute impurity using the configured criterion."""
        if self.criterion == "entropy":
            return self._entropy(y)
        return self._gini(y)

    @staticmethod
    def _entropy(y: np.ndarray) -> float:
        """Compute Shannon entropy: -Σ p_i × log2(p_i)."""
        if len(y) == 0:
            return 0.0
        # Offset labels so the minimum is 0 (np.bincount requires non-negative ints)
        counts = np.bincount(y - y.min())
        probs = counts[counts > 0] / len(y)
        return float(-np.sum(probs * np.log2(probs)))

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        """Compute Gini impurity: 1 - Σ p_i²."""
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y - y.min())
        probs = counts[counts > 0] / len(y)
        return float(1.0 - np.sum(probs ** 2))

    @staticmethod
    def _most_common_label(y: np.ndarray) -> int:
        """Return the most frequent label in the array."""
        # Offset then restore so argmax returns the actual label value
        offset = y.min()
        return int(np.bincount(y - offset).argmax() + offset)

    # ================================================================
    # Prediction (Private)
    # ================================================================

    def _traverse(self, x: np.ndarray, node: Node) -> int:
        """Recursively traverse the tree to predict a single sample."""
        if node.is_leaf:
            return node.value  # type: ignore[return-value]

        if x[node.feature_index] <= node.threshold:
            return self._traverse(x, node.left)  # type: ignore[arg-type]
        return self._traverse(x, node.right)  # type: ignore[arg-type]

    # ================================================================
    # Tree Inspection (Private)
    # ================================================================

    @staticmethod
    def _compute_depth(node: Node) -> int:
        """Compute the maximum depth from a given node."""
        if node.is_leaf:
            return 0
        left_depth = DecisionTree._compute_depth(node.left) if node.left else 0
        right_depth = DecisionTree._compute_depth(node.right) if node.right else 0
        return 1 + max(left_depth, right_depth)

    @staticmethod
    def _count_leaves(node: Node) -> int:
        """Count the total number of leaf nodes below a given node."""
        if node.is_leaf:
            return 1
        left_count = DecisionTree._count_leaves(node.left) if node.left else 0
        right_count = DecisionTree._count_leaves(node.right) if node.right else 0
        return left_count + right_count
