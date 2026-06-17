#!/usr/bin/env python3
"""Generate golden test data for Go XGBoost implementation validation."""

import json, os, sys
import numpy as np
import xgboost as xgb

# Output directory relative to this script's location: ../../testdata/
# (script is at lib/xgb/tools/gen_golden_data.py → target is lib/xgb/testdata/)
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "testdata"))
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_regression():
    """Simple linear regression with noise."""
    np.random.seed(42)
    n, m = 200, 5
    X = np.random.randn(n, m)
    y = 2*X[:, 0] - X[:, 1] + 0.5*X[:, 2] + np.random.randn(n)*0.1

    params = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',  # must match Go histogram impl
        'max_depth': 4,
        'eta': 0.3,
        'lambda': 1.0,
        'gamma': 0.0,
        'seed': 42,
        'verbosity': 0,
    }
    dtrain = xgb.DMatrix(X, label=y)
    model = xgb.train(params, dtrain, num_boost_round=10)
    pred = model.predict(dtrain)

    save("regression_basic", X, y, pred, model, params)
    print(f"  regression_basic: {n}s x {m}f, RMSE={np.sqrt(np.mean((pred-y)**2)):.4f}")


def generate_classification():
    """Linearly separable binary classification."""
    np.random.seed(42)
    n, m = 300, 5
    X = np.random.randn(n, m)
    y = (X[:, 0] + X[:, 1] > 0).astype(float)

    params = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',  # must match Go histogram impl
        'max_depth': 4,
        'eta': 0.3,
        'lambda': 1.0,
        'gamma': 0.0,
        'seed': 42,
        'verbosity': 0,
    }
    dtrain = xgb.DMatrix(X, label=y)
    model = xgb.train(params, dtrain, num_boost_round=10)
    pred = model.predict(dtrain)

    save("classification_basic", X, y, pred, model, params)
    acc = np.mean((pred > 0.5).astype(float) == y)
    print(f"  classification_basic: {n}s x {m}f, accuracy={acc:.4f}")


def generate_small():
    """Very small dataset for quick unit tests."""
    np.random.seed(1)
    n, m = 20, 3
    X = np.random.randn(n, m)
    y = X[:, 0] * 2 + X[:, 1]

    params = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',  # must match Go histogram impl
        'max_depth': 3,
        'eta': 0.5,
        'lambda': 1.0,
        'gamma': 0.0,
        'seed': 1,
        'verbosity': 0,
    }
    dtrain = xgb.DMatrix(X, label=y)
    model = xgb.train(params, dtrain, num_boost_round=5)
    pred = model.predict(dtrain)

    save("small_regression", X, y, pred, model, params)
    print(f"  small_regression: {n}s x {m}f")


def generate_with_missing():
    """Binary classification with missing values (NaN)."""
    np.random.seed(42)
    n, m = 200, 5
    X = np.random.randn(n, m)
    # Randomly set 10% of values to NaN
    mask = np.random.random(X.shape) < 0.1
    X[mask] = np.nan
    y = (np.nanmean(X[:, 0:2], axis=1) > 0).astype(float)

    params = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',  # must match Go histogram impl
        'max_depth': 4,
        'eta': 0.3,
        'lambda': 1.0,
        'seed': 42,
        'verbosity': 0,
    }
    dtrain = xgb.DMatrix(X, label=y)
    model = xgb.train(params, dtrain, num_boost_round=10)
    pred = model.predict(dtrain)

    save("classification_missing", X, y, pred, model, params)
    print(f"  classification_missing: {n}s x {m}f, {np.sum(mask)} NaN values")


def generate_deep():
    """Deeper model with more features."""
    np.random.seed(42)
    n, m = 500, 10
    X = np.random.randn(n, m)
    y = (X[:, 0]*X[:, 1] + X[:, 2] + np.sin(X[:, 3]) + np.random.randn(n)*0.2 > 0.5).astype(float)

    params = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',  # must match Go histogram impl
        'max_depth': 8,
        'eta': 0.1,
        'lambda': 2.0,
        'gamma': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42,
        'verbosity': 0,
    }
    dtrain = xgb.DMatrix(X, label=y)
    model = xgb.train(params, dtrain, num_boost_round=50)
    pred = model.predict(dtrain)

    save("classification_deep", X, y, pred, model, params)
    acc = np.mean((pred > 0.5).astype(float) == y)
    print(f"  classification_deep: {n}s x {m}f, accuracy={acc:.4f}")


def save(name, X, y, pred, model, params):
    """Save test data to CSV and JSON files."""
    prefix = os.path.join(OUTPUT_DIR, name)

    # Save features and labels with full float32 precision
    np.savetxt(f"{prefix}_features.csv", X, delimiter=",", fmt="%.18f")
    np.savetxt(f"{prefix}_labels.csv", y, delimiter=",", fmt="%.18f")
    np.savetxt(f"{prefix}_pred.csv", pred, delimiter=",", fmt="%.18f")

    # Save model config used
    with open(f"{prefix}_config.json", "w") as f:
        json.dump(params, f, indent=2)

    # Save tree structure (dump)
    model.dump_model(f"{prefix}_dump.json", with_stats=True, dump_format="json")

    # Save full model
    model.save_model(f"{prefix}_model.json")


if __name__ == "__main__":
    print("Generating golden test data...")
    generate_regression()
    generate_classification()
    generate_small()
    generate_with_missing()
    generate_deep()
    print(f"\n✓ All data saved to {OUTPUT_DIR}/")
