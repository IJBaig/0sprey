"""
ML Pipeline for Network Intrusion Detection.
- Random Forest (high accuracy) + SGD (true incremental learning)
- ONNX as primary model format (cross-platform)
"""
import os
import gc
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
import sklearn
import warnings

warnings.filterwarnings("ignore")

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import onnxruntime as rt
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install skl2onnx onnxruntime")
    raise

DROP_COLUMNS = [
    "Flow ID", "Source IP", "Src IP", "Source Port",
    "Destination IP", "Dst IP",
    "Timestamp", "Label", "label",
]

ALGORITHMS = {
    "rf": "Random Forest",
    "sgd": "SGD Classifier",
}


# ══════════════════════════════════════════════════════════════
#  DATA PROCESSING
# ══════════════════════════════════════════════════════════════

def _clean_data(df):
    """Clean and preprocess the CICFlowMeter feature DataFrame."""
    df.columns = df.columns.str.strip()

    label_col = None
    for col in df.columns:
        if col.lower() == "label":
            label_col = col
            break

    if label_col is None:
        raise ValueError("No 'Label' column found in CSV.")

    labels = df[label_col].copy()
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    X = df.drop(columns=cols_to_drop, errors="ignore")
    X = X.select_dtypes(include=[np.number])
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    return X, labels


def _encode_labels(y):
    """Encode string labels to integers."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))
    print(f"\n[*] Classes found ({len(le.classes_)}):")
    for i, cls in enumerate(le.classes_):
        count = np.sum(y_encoded == i)
        print(f"    {i}: {cls} ({count} samples)")
    return y_encoded, le


def _get_model(algo):
    """Initialize the ML model."""
    if algo == "rf":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            n_jobs=-1,
            random_state=42,
            warm_start=True,
            class_weight="balanced",
        )
    elif algo == "sgd":
        return SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.0001,
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}. Choose 'rf' or 'sgd'.")


def _load_csv(csv_path):
    """Load CSV with memory optimization."""
    file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"      File size: {file_size_mb:.1f} MB")

    df = pd.read_csv(csv_path, low_memory=False)
    print(f"      Loaded: {len(df):,} rows, {len(df.columns)} columns")

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"      Memory usage: {mem_mb:.1f} MB")

    return df


def _save_onnx(model, meta, onnx_path):
    """Convert sklearn model to ONNX and save with metadata."""
    n_features = len(meta["feature_names"])
    initial_type = [("input", FloatTensorType([None, n_features]))]

    onnx_model = convert_sklearn(model, initial_types=initial_type)

    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)

    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    meta_path = onnx_path.replace(".onnx", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    size_model = os.path.getsize(onnx_path) / (1024 * 1024)
    size_meta = os.path.getsize(meta_path) / 1024
    print(f"  ✅ ONNX model:  {onnx_path} ({size_model:.1f} MB)")
    print(f"  ✅ Metadata:    {meta_path} ({size_meta:.1f} KB)")


def _load_onnx(onnx_path):
    """Load ONNX model and its metadata."""
    meta_path = onnx_path.replace(".onnx", "_meta.json")

    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    session = rt.InferenceSession(onnx_path)

    with open(meta_path, "r") as f:
        meta = json.load(f)

    return session, meta


# ══════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════

def train_model(csv_path, model_path, algo="rf", test_size=0.2):
    """
    Train an ML model and save as ONNX.

    Args:
        csv_path:   Path to labeled CSV file (with 'Label' column)
        model_path: Output path for .onnx model
        algo:       'rf' or 'sgd'
        test_size:  Fraction of data for testing (default 0.2)
    """
    print(f"{'='*60}")
    print(f"  IDS Model Training Pipeline")
    print(f"{'='*60}")
    print(f"  Algorithm : {ALGORITHMS.get(algo, algo)}")
    print(f"  Input     : {csv_path}")
    print(f"  Output    : {model_path}")
    print(f"{'='*60}")

    # ── Load Data ─────────────────────────────────────────────
    print(f"\n[1/6] Loading data...")
    df = _load_csv(csv_path)
    gc.collect()

    # ── Clean & Preprocess ────────────────────────────────────
    print(f"\n[2/6] Cleaning data...")
    X, y = _clean_data(df)
    del df
    gc.collect()
    print(f"      Features: {X.shape[1]}, Samples: {X.shape[0]}")

    # ── Encode Labels ─────────────────────────────────────────
    print(f"\n[3/6] Encoding labels...")
    y_encoded, label_encoder = _encode_labels(y)
    del y
    gc.collect()

    # ── Scale Features ────────────────────────────────────────
    feature_names = list(X.columns)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    del X
    gc.collect()

    # ── Split Data ────────────────────────────────────────────
    print(f"\n[4/6] Splitting data (test_size={test_size})...")

    unique, counts = np.unique(y_encoded, return_counts=True)
    rare_classes = unique[counts < 2]
    if len(rare_classes) > 0:
        rare_labels = label_encoder.inverse_transform(rare_classes)
        print(f"      ⚠ Removing classes with <2 samples: {list(rare_labels)}")
        mask = ~np.isin(y_encoded, rare_classes)
        X_scaled = X_scaled[mask]
        y_encoded = y_encoded[mask]
        print(f"      Remaining samples: {len(y_encoded):,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded,
        test_size=test_size,
        random_state=42,
        stratify=y_encoded,
    )
    del X_scaled, y_encoded
    gc.collect()
    print(f"      Train: {len(X_train):,}, Test: {len(X_test):,}")

    # ── Train Model ───────────────────────────────────────────
    print(f"\n[5/6] Training {ALGORITHMS.get(algo, algo)}...")
    model = _get_model(algo)

    if algo == "sgd":
        classes = np.unique(y_train)
        sample_weights = compute_sample_weight("balanced", y_train)
        n_epochs = 10
        for epoch in range(1, n_epochs + 1):
            indices = np.random.RandomState(epoch).permutation(len(X_train))
            model.partial_fit(
                X_train[indices],
                y_train[indices],
                classes=classes,
                sample_weight=sample_weights[indices],
            )
            if epoch % 2 == 0 or epoch == n_epochs:
                acc_e = accuracy_score(y_test, model.predict(X_test))
                print(f"      Epoch {epoch}/{n_epochs} — Accuracy: {acc_e:.4f}")
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"\n{'─'*60}")
    print(f"  RESULTS")
    print(f"{'─'*60}")
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")

    present_labels = np.unique(np.concatenate([y_test, y_pred]))
    present_names = label_encoder.inverse_transform(present_labels)

    print(classification_report(
        y_test, y_pred, labels=present_labels,
        target_names=present_names, zero_division=0,
    ))

    if algo == "rf":
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        print(f"  Top 10 Important Features:")
        for rank, idx in enumerate(indices, 1):
            print(f"    {rank:2d}. {feature_names[idx]:<35s} {importances[idx]:.4f}")

    # ── Export to ONNX ────────────────────────────────────────
    print(f"\n[6/6] Exporting to ONNX...")

    meta = {
        "feature_names": feature_names,
        "classes": list(label_encoder.classes_),
        "algorithm": algo,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "sklearn_version": sklearn.__version__,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_samples": int(len(X_train)),
        "accuracy": float(acc),
        "update_count": 0,
    }

    _save_onnx(model, meta, model_path)

    print(f"\n{'─'*60}")
    print(f"  📦 Model ready for deployment on any machine")
    print(f"  📋 Requirements: pip install onnxruntime pandas numpy")
    print(f"{'─'*60}")


# ══════════════════════════════════════════════════════════════
#  INCREMENTAL TRAINING
# ══════════════════════════════════════════════════════════════

def update_model(model_path, new_csv_path, algo=None):
    """
    Update model with new labeled data without retraining from scratch.

    Args:
        model_path:   Path to existing .onnx model
        new_csv_path: Path to new labeled CSV data
        algo:         Override algorithm (None = use original)
    """
    print(f"{'='*60}")
    print(f"  IDS Model — Incremental Update")
    print(f"{'='*60}")

    # ── Load Existing Metadata
    print(f"\n[1/5] Loading existing model metadata...")
    meta_path = model_path.replace(".onnx", "_meta.json")

    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    if algo is None:
        algo = meta["algorithm"]

    feature_names = meta["feature_names"]
    old_classes = meta["classes"]
    update_count = meta.get("update_count", 0)
    scaler_mean = np.array(meta["scaler_mean"])
    scaler_scale = np.array(meta["scaler_scale"])

    print(f"      Algorithm: {ALGORITHMS.get(algo, algo)}")
    print(f"      Previous updates: {update_count}")
    print(f"      Existing classes: {old_classes}")

    # Rebuild scaler from metadata
    scaler = StandardScaler()
    scaler.mean_ = scaler_mean
    scaler.scale_ = scaler_scale
    scaler.var_ = scaler_scale ** 2
    scaler.n_features_in_ = len(feature_names)

    # Rebuild label encoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(old_classes)

    # ── Load New Data 
    print(f"\n[2/5] Loading new data...")
    df = _load_csv(new_csv_path)

    print(f"\n[3/5] Processing new data...")
    X_new, y_new = _clean_data(df)
    del df
    gc.collect()

    X_new = X_new.reindex(columns=feature_names, fill_value=0)
    print(f"      New samples: {X_new.shape[0]:,}, Features: {X_new.shape[1]}")

    # Handle new classes
    known_classes = set(label_encoder.classes_)
    new_classes = set(y_new.unique()) - known_classes
    if new_classes:
        print(f"      ⚠ New classes found: {new_classes}")
        print(f"        Adding them to the label encoder...")
        all_classes = list(label_encoder.classes_) + sorted(new_classes)
        label_encoder.classes_ = np.array(all_classes)

    y_new_encoded = label_encoder.transform(y_new.astype(str))

    X_new_values = X_new.values
    scaler.partial_fit(X_new_values)
    X_new_scaled = scaler.transform(X_new_values)
    del X_new, X_new_values
    gc.collect()

    # ── Train Updated Model 
    print(f"\n[4/5] Training updated model...")
    model = _get_model(algo)

    if algo == "sgd":
        all_classes_idx = np.arange(len(label_encoder.classes_))
        sample_weights = compute_sample_weight("balanced", y_new_encoded)
        n_epochs = 10
        for epoch in range(1, n_epochs + 1):
            indices = np.random.RandomState(epoch).permutation(len(X_new_scaled))
            model.partial_fit(
                X_new_scaled[indices],
                y_new_encoded[indices],
                classes=all_classes_idx,
                sample_weight=sample_weights[indices],
            )
            if epoch % 2 == 0 or epoch == n_epochs:
                acc_e = accuracy_score(y_new_encoded, model.predict(X_new_scaled))
                print(f"      Epoch {epoch}/{n_epochs} — Accuracy: {acc_e:.4f}")
        print(f"      ✅ SGD: Incremental update via partial_fit()")
    elif algo == "rf":
        model.fit(X_new_scaled, y_new_encoded)
        print(f"      ✅ RF: Retrained with new data ({model.n_estimators} trees)")

    # ── Export Updated ONNX 
    print(f"\n[5/5] Exporting updated ONNX...")

    updated_meta = {
        "feature_names": feature_names,
        "classes": list(label_encoder.classes_),
        "algorithm": algo,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "sklearn_version": sklearn.__version__,
        "trained_at": meta.get("trained_at", "unknown"),
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_samples": meta.get("training_samples", 0),
        "last_update_samples": int(len(y_new_encoded)),
        "update_count": update_count + 1,
    }

    _save_onnx(model, updated_meta, model_path)

    print(f"\n{'─'*60}")
    print(f"  📊 Total updates: {updated_meta['update_count']}")
    print(f"{'─'*60}")


# ══════════════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════════════

def predict(model_path, df):
    """
    Run prediction on a pandas DataFrame using ONNX model.

    Args:
        model_path: Path to .onnx model
        df:         pandas DataFrame with CICFlowMeter features

    Returns:
        DataFrame with 'label' column added containing predictions
    """
    print(f"[*] Loading ONNX model: {model_path}")
    session, meta = _load_onnx(model_path)

    feature_names = meta["feature_names"]
    classes = meta["classes"]
    scaler_mean = np.array(meta["scaler_mean"])
    scaler_scale = np.array(meta["scaler_scale"])

    print(f"    Algorithm: {ALGORITHMS.get(meta['algorithm'], meta['algorithm'])}")
    print(f"    Features:  {len(feature_names)}")
    print(f"    Classes:   {classes}")
    print(f"    Updates:   {meta.get('update_count', 0)}")

    df = df.copy()
    df.columns = df.columns.str.strip()

    X = df.reindex(columns=feature_names, fill_value=0)
    X = X.select_dtypes(include=[np.number])
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    X_scaled = (X.values - scaler_mean) / scaler_scale
    X_scaled = X_scaled.astype(np.float32)

    input_name = session.get_inputs()[0].name
    pred = session.run(None, {input_name: X_scaled})[0]

    df["label"] = [classes[p] for p in pred]

    print(f"\n[*] Predictions on {len(df):,} flows:")
    print(df["label"].value_counts().to_string())

    return df
