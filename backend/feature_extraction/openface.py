

import subprocess
from pathlib import Path

import pandas as pd


GAZE_COLS = [
    "gaze_0_x", "gaze_0_y", "gaze_0_z",
    "gaze_1_x", "gaze_1_y", "gaze_1_z",
    "gaze_angle_x", "gaze_angle_y",
]

POSE_COLS = [
    "pose_Tx", "pose_Ty", "pose_Tz",
    "pose_Rx", "pose_Ry", "pose_Rz",
]

AU_INTENSITY_COLS = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r",
    "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r",
    "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r",
    "AU26_r", "AU45_r",
]

AU_OCCURRENCE_COLS = [
    "AU01_c", "AU02_c", "AU04_c", "AU05_c", "AU06_c",
    "AU07_c", "AU09_c", "AU10_c", "AU12_c", "AU14_c",
    "AU15_c", "AU17_c", "AU20_c", "AU23_c", "AU25_c",
    "AU26_c", "AU28_c", "AU45_c",
]

OPENFACE_FEATURE_COLS = GAZE_COLS + POSE_COLS + AU_INTENSITY_COLS

CONFIDENCE_THRESHOLD = 0.8

# Docker image name built locally — matches what we called it during setup
DOCKER_IMAGE = "my-openface:latest"
FEATURE_EXTRACTION_BIN = "/opt/OpenFace/build/bin/FeatureExtraction"


def run_openface_docker(video_path: Path, output_dir: Path, timeout: int = 300) -> Path:
    """Run OpenFace FeatureExtraction on a video via Docker.

    Args:
        video_path: Path to input video on the host
        output_dir: Host directory where OpenFace writes its CSV output
        timeout: Max seconds to wait for OpenFace (default 5 min)

    Returns:
        Path to the per-frame CSV produced by OpenFace.

    Raises:
        RuntimeError: If the Docker subprocess fails or no output is produced.
    """
    video_path = video_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    video_dir = video_path.parent
    video_name = video_path.name

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{video_dir}:/in",
        "-v", f"{output_dir}:/out",
        DOCKER_IMAGE,
        FEATURE_EXTRACTION_BIN,
        "-f", f"/in/{video_name}",
        "-out_dir", "/out",
        "-aus",     # extract action units
        "-gaze",    # extract gaze (required — training pool includes it)
        "-pose",    # extract head pose
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    if result.returncode != 0:
        raise RuntimeError(
            f"OpenFace Docker failed (exit {result.returncode}):\n"
            f"stderr: {result.stderr[-2000:]}"
        )

    csv_path = output_dir / f"{video_path.stem}.csv"
    if not csv_path.exists():
        raise RuntimeError(
            f"OpenFace completed but no CSV at {csv_path}. "
            f"stdout tail: {result.stdout[-500:]}"
        )

    return csv_path


def aggregate_openface(csv_path: Path) -> dict:
    """Aggregate per-frame OpenFace output into a single feature dict.

    Mirrors the training-time aggregation in Feature_Extraction_OpenFace.ipynb:
        - Filter by confidence > 0.8
        - mean and std for gaze, pose, and AU intensity columns
        - activation rate (mean) for AU occurrence columns

    Args:
        csv_path: Path to OpenFace's per-frame CSV

    Returns:
        Dict mapping feature_name -> float. Keys match training columns:
            {gaze_0_x}_mean, {gaze_0_x}_std, ..., {AU01_c}_rate, ...
    """
    df = pd.read_csv(csv_path)

    # Strip whitespace from column names (OpenFace adds leading spaces)
    df.columns = df.columns.str.strip()

    # Filter to high-confidence frames
    df = df[df["confidence"] > CONFIDENCE_THRESHOLD]

    if len(df) == 0:
        raise RuntimeError(
            f"No frames passed confidence threshold {CONFIDENCE_THRESHOLD} "
            "Face detection may have failed."
        )

    features = {}

    for col in OPENFACE_FEATURE_COLS:
        if col not in df.columns:
            raise RuntimeError(f"Expected column '{col}' missing from OpenFace CSV")
        features[f"{col}_mean"] = float(df[col].mean())
        features[f"{col}_std"]  = float(df[col].std())

    # AU occurrence: activation rate (mean is the proportion of frames active)
    for col in AU_OCCURRENCE_COLS:
        if col not in df.columns:
            raise RuntimeError(f"Expected column '{col}' missing from OpenFace CSV")
        features[f"{col}_rate"] = float(df[col].mean())

    return features
