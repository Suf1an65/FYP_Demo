"""Compute head motion kinematic features from OpenFace per-frame rotation data.

These features are additions to the raw OpenFace output, matching the
training-time `add_features_cell` logic:
    - pose_R{x,y,z}_total_motion:  sum of |Δrotation| across frames
    - pose_R{x,y,z}_mean_velocity: mean |Δrotation| per frame
    - pose_R{x,y,z}_jerk:          mean |Δ²rotation| per frame (jitteriness)
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROTATION_COLS = ["pose_Rx", "pose_Ry", "pose_Rz"]
CONFIDENCE_THRESHOLD = 0.8


def compute_head_motion(csv_path: Path) -> dict:
    """Compute head motion features from OpenFace's per-frame CSV.

    Args:
        csv_path: Path to OpenFace's per-frame CSV

    Returns:
        Dict with 9 features: 3 metrics × 3 rotation axes.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Match training-time confidence filter
    df = df[df["confidence"] > CONFIDENCE_THRESHOLD].reset_index(drop=True)

    if len(df) < 3:
        raise RuntimeError(
            f"Not enough high-confidence frames ({len(df)}) to compute head motion "
            "(need at least 3 for second derivative)."
        )

    features = {}
    for col in ROTATION_COLS:
        if col not in df.columns:
            raise RuntimeError(f"Expected rotation column '{col}' missing from OpenFace CSV")

        signal = df[col].values
        velocity = np.diff(signal)
        acceleration = np.diff(velocity)

        features[f"{col}_total_motion"]  = float(np.sum(np.abs(velocity)))
        features[f"{col}_mean_velocity"] = float(np.mean(np.abs(velocity)))
        features[f"{col}_jerk"]          = float(np.mean(np.abs(acceleration)))

    return features
