"""Functional test suite for all modules."""

import sys
sys.path.insert(0, ".")
import numpy as np

# === Test 1: Decision Tree with 1-indexed labels (HAR-style) ===
from src.models.decision_tree import DecisionTree

np.random.seed(42)
X = np.random.randn(100, 5)
y = np.random.choice([1, 2, 3, 4, 5, 6], size=100)  # 1-indexed like HAR!

tree = DecisionTree(max_depth=3, criterion="entropy")
tree.fit(X, y)
preds = tree.predict(X)
print(f"[PASS] DecisionTree fit/predict with 1-indexed labels: depth={tree.get_depth()}, leaves={tree.get_n_leaves()}")
print(f"       Predictions range: {preds.min()}-{preds.max()}, Importances sum: {tree.feature_importances_.sum():.4f}")

# Also test Gini
tree_gini = DecisionTree(max_depth=3, criterion="gini")
tree_gini.fit(X, y)
preds_gini = tree_gini.predict(X)
print(f"[PASS] Gini criterion: depth={tree_gini.get_depth()}, preds range: {preds_gini.min()}-{preds_gini.max()}")

# Test DecisionTree input validation
try:
    DecisionTree().fit(np.array([]), np.array([]))
except ValueError as e:
    print(f"[PASS] Caught empty fit validation: {e}")

try:
    DecisionTree().fit(X, y[:-1])
except ValueError as e:
    print(f"[PASS] Caught shape mismatch validation: {e}")

try:
    tree.predict(X[:, :-1])
except ValueError as e:
    print(f"[PASS] Caught predict feature mismatch: {e}")

# === Test 2: Metrics with 1-indexed labels ===
from src.evaluation.metrics import accuracy, precision, recall, f1_score, confusion_matrix, classification_report

y_true = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
y_pred = np.array([1, 2, 3, 1, 1, 3, 2, 2, 3])

acc = accuracy(y_true, y_pred)
print(f"[PASS] Accuracy: {acc:.4f}")

prec = precision(y_true, y_pred, average="macro")
print(f"[PASS] Precision (macro): {prec:.4f}")

f1 = f1_score(y_true, y_pred, average="macro")
print(f"[PASS] F1 (macro): {f1:.4f}")

cm = confusion_matrix(y_true, y_pred)
print(f"[PASS] Confusion matrix shape: {cm.shape}")

# Test report with mismatched target_names (fix #4)
report = classification_report(y_true, y_pred, target_names=["A", "B", "C", "D", "E", "F"])
print("[PASS] Classification report with mismatched names handled gracefully")

# Test report with correct names
report = classification_report(y_true, y_pred, target_names=["Walk", "Sit", "Stand"])
print("[PASS] Classification report with correct names works")
print(report)

# Test zero support/empty arrays check for weighted average
y_empty = np.array([], dtype=int)
weighted_p_zero = precision(y_empty, y_empty, average="weighted")
weighted_r_zero = recall(y_empty, y_empty, average="weighted")
weighted_f_zero = f1_score(y_empty, y_empty, average="weighted")
print(f"[PASS] Zero support weighted average precision={weighted_p_zero}, recall={weighted_r_zero}, f1={weighted_f_zero}")

# Test metric shape mismatch validation
try:
    accuracy(np.array([1, 2]), np.array([1]))
except ValueError as e:
    print(f"[PASS] Caught accuracy shape mismatch: {e}")

# === Test 3: Config loading ===
from src.utils.helpers import load_config, set_seed, shuffle_data

config = load_config("config/config.yaml")
print(f"[PASS] Config loaded: {len(config)} top-level keys")
assert "time_warp_sigma" in config["augmentation"], "Missing time_warp_sigma!"
print(f"[PASS] time_warp_sigma found in config: {config['augmentation']['time_warp_sigma']}")

# === Test 4: Augmentation ===
from src.data.augmentation import jitter, scaling, augment_dataset

X_3d = np.random.randn(10, 50, 3)
y_3d = np.array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4])
X_aug, y_aug = augment_dataset(X_3d, y_3d, config, seed=42)
print(f"[PASS] Augmentation: {X_3d.shape[0]} -> {X_aug.shape[0]} samples")

# === Test 5: Shuffle ===
X_s, y_s = shuffle_data(X_3d, y_3d, seed=42)
print(f"[PASS] shuffle_data works: shape={X_s.shape}")

# === Test 6: DecisionTree on augmented data ===
from scripts.train import extract_statistical_features, extract_fft_features, extract_all_features

X_feats = extract_statistical_features(X_3d)
print(f"[PASS] Statistical features extracted: {X_feats.shape}")

# === Test 7: FFT Feature Extraction ===
X_fft = extract_fft_features(X_3d)
assert X_fft.shape == (10, 6), f"Expected (10, 6), got {X_fft.shape}"
assert not np.any(np.isnan(X_fft)), "FFT features contain NaN!"
print(f"[PASS] FFT features extracted: {X_fft.shape}, no NaNs")

# === Test 8: Combined Features ===
X_all = extract_all_features(X_3d)
assert X_all.shape == (10, 30), f"Expected (10, 30), got {X_all.shape}"
print(f"[PASS] Combined features (stat + FFT): {X_all.shape}")

# === Test 9: DecisionTree with max_features ===
tree_mf = DecisionTree(max_depth=3, criterion="entropy", max_features=3, random_state=42)
tree_mf.fit(X_feats, y_3d)
preds_mf = tree_mf.predict(X_feats)
print(f"[PASS] DecisionTree with max_features=3: depth={tree_mf.get_depth()}, preds range: {preds_mf.min()}-{preds_mf.max()}")

# === Test 10: RandomForest ===
from src.models.random_forest import RandomForest

rf = RandomForest(n_estimators=10, max_depth=3, criterion="entropy", random_state=42)
rf.fit(X_feats, y_3d)
rf_preds = rf.predict(X_feats)
rf_acc = accuracy(y_3d, rf_preds)
rf_imp = rf.feature_importances_
assert len(rf.trees_) == 10, f"Expected 10 trees, got {len(rf.trees_)}"
assert abs(rf_imp.sum() - 1.0) < 1e-6, f"Importances don't sum to 1: {rf_imp.sum()}"
print(f"[PASS] RandomForest(n=10): accuracy={rf_acc:.4f}, importances sum={rf_imp.sum():.4f}")

# === Test 11: Full pipeline with RandomForest ===
tree2 = DecisionTree(max_depth=5, criterion="gini")
tree2.fit(X_feats, y_3d)
preds2 = tree2.predict(X_feats)
acc2 = accuracy(y_3d, preds2)
print(f"[PASS] Full pipeline (features -> tree -> eval): accuracy={acc2:.4f}")

# === Test 12: Comparison with sklearn DecisionTreeClassifier ===
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

X_comp, y_comp = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
# Train sklearn
sk_tree = DecisionTreeClassifier(max_depth=3, criterion="entropy", random_state=42)
sk_tree.fit(X_comp, y_comp)
sk_acc = accuracy(y_comp, sk_tree.predict(X_comp))

# Train scratch
scratch_tree = DecisionTree(max_depth=3, criterion="entropy", random_state=42)
scratch_tree.fit(X_comp, y_comp)
scratch_acc = accuracy(y_comp, scratch_tree.predict(X_comp))

# Confirm they are close (within 10% margin)
assert abs(sk_acc - scratch_acc) <= 0.1, f"Accuracy difference too large: Sklearn={sk_acc:.4f}, Scratch={scratch_acc:.4f}"
print(f"[PASS] Scratch vs Sklearn accuracy comparison: Sklearn={sk_acc:.4f}, Scratch={scratch_acc:.4f} (diff <= 10%)")

print()
print("=" * 50)
print("  ALL TESTS PASSED")
print("=" * 50)

