import json
from pathlib import Path

import joblib
import numpy as np


# ── Locate model artefacts ─────────────────────────────────────────
# Resolve relative to this file so it works regardless of CWD
_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent
_MODEL_DIR = _PROJECT_ROOT / "model"

SVM_PATH    = _MODEL_DIR / "svm_model.joblib"
CONFIG_PATH = _MODEL_DIR / "deployment_config.json"

if not SVM_PATH.exists():
    raise FileNotFoundError(
        f"SVM model not found at {SVM_PATH}. "
        "Did you download svm_model.joblib from Colab into backend/model/?"
    )
if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Deployment config not found at {CONFIG_PATH}. "
        "Did you download deployment_config.json from Colab into backend/model/?"
    )


_SVM = joblib.load(SVM_PATH)

with open(CONFIG_PATH) as f:
    _CONFIG = json.load(f)

# Pre-compute numpy arrays for fast inference
_ALL_COLUMNS = _CONFIG["all_feature_columns"]
_SELECTED_INDICES = np.array(_CONFIG["selected_indices"])
_BASELINES_NUMPY = {
    valence: {
        "mu":    np.array(_CONFIG["baselines"][valence]["mu"]),
        "sigma": np.array(_CONFIG["baselines"][valence]["sigma"]),
    }
    for valence in _CONFIG["baselines"]
}

print(f"[inference] Loaded SVM with {len(_CONFIG['selected_features'])} features")
print(f"[inference] Feature pool: {len(_ALL_COLUMNS)} columns")
print(f"[inference] Valence baselines: {list(_CONFIG['baselines'].keys())}")


def predict(features: dict, valence: str) -> dict:
    """Predict deception from extracted features.

    Args:
        features: Dict mapping feature name -> float, with all 188 keys
                  matching the training schema (produced by feature_extraction).
        valence:  'positive' or 'negative' — the prompt category the subject
                  was responding to. Set in the frontend, not inferred.

    Returns:
        Dict with:
            prediction (str): 'lie' or 'truth'
            confidence (float): probability of the predicted class (0.0-1.0)
            raw_probabilities (dict): both class probabilities for reference

    Raises:
        ValueError: If valence isn't recognised, or if features is missing
                    expected keys.
    """
    # ── Validate valence ───────────────────────────────────────────
    if valence not in _BASELINES_NUMPY:
        raise ValueError(
            f"Unknown valence '{valence}'. "
            f"Expected one of: {list(_BASELINES_NUMPY.keys())}"
        )

    # ── Validate feature dict has all expected keys ────────────────
    missing = [c for c in _ALL_COLUMNS if c not in features]
    if missing:
        raise ValueError(
            f"Feature dict missing {len(missing)} expected keys. "
            f"First few: {missing[:5]}"
        )

    # ── Build feature vector in canonical column order ─────────────
    x_raw = np.array([float(features[col]) for col in _ALL_COLUMNS])

    # ── Apply valence-matched z-score normalisation ────────────────
    baseline = _BASELINES_NUMPY[valence]
    x_norm = (x_raw - baseline["mu"]) / baseline["sigma"]

    # ── Select the features the SVM was trained on ────────────────
    x_selected = x_norm[_SELECTED_INDICES]

    # ── Predict ────────────────────────────────────────────────────
    pred_label = int(_SVM.predict([x_selected])[0])
    probs = _SVM.predict_proba([x_selected])[0]

    # The SVM's classes_ attribute tells us the label-to-index mapping
    classes = _SVM.classes_
    prob_lie   = float(probs[list(classes).index(1)])
    prob_truth = float(probs[list(classes).index(0)])

    prediction = "lie" if pred_label == 1 else "truth"
    confidence = prob_lie if pred_label == 1 else prob_truth

    return {
        "prediction": prediction,
        "confidence": confidence,
        "raw_probabilities": {
            "truth": prob_truth,
            "lie":   prob_lie,
        },
    }