"""End-to-end pipeline test: video in, prediction out.

Runs feature extraction + valence normalisation + SVM prediction on a
single video. Confirms the full inference path works before wiring into
FastAPI.

Usage:
    python test_inference.py <video_path> <valence>

Example:
    python test_inference.py C:\\Users\\masuf\\Downloads\\BF001_1PT.mp4 positive
"""

import sys
import time
from pathlib import Path

from feature_extraction.audio import extract_audio
from feature_extraction.openface import aggregate_openface, run_openface_docker
from feature_extraction.head_motion import compute_head_motion
from feature_extraction.opensmile_wrapper import extract_opensmile
from feature_extraction.transcript import transcribe
from feature_extraction.linguistic import extract_linguistic
from feature_extraction.inference import predict


def main(video_path_str: str, valence: str):
    if valence not in ("positive", "negative"):
        print(f"ERROR: valence must be 'positive' or 'negative', got '{valence}'")
        sys.exit(1)

    video_path = Path(video_path_str)
    if not video_path.exists():
        print(f"ERROR: Video not found at {video_path}")
        sys.exit(1)

    work_dir = Path("./test_extraction_output")
    work_dir.mkdir(exist_ok=True)

    print(f"Video:   {video_path}")
    print(f"Valence: {valence}")
    print()

    pipeline_start = time.time()

    # ── Extract features ───────────────────────────────────────────
    print("[1/5] OpenFace via Docker...")
    csv_path = run_openface_docker(video_path, work_dir / "openface")
    visual = aggregate_openface(csv_path)
    head_motion = compute_head_motion(csv_path)

    print("[2/5] Audio extraction...")
    audio_path = extract_audio(video_path, work_dir / "audio")

    print("[3/5] OpenSMILE...")
    audio_features = extract_opensmile(audio_path)

    print("[4/5] Whisper transcription...")
    transcript = transcribe(audio_path)

    print("[5/5] Linguistic features...")
    linguistic = extract_linguistic(transcript)

    all_features = {
        **visual,
        **head_motion,
        **audio_features,
        **linguistic,
    }

    extraction_elapsed = time.time() - pipeline_start
    print(f"\n✓ Extraction complete: {len(all_features)} features in {extraction_elapsed:.1f}s")

    # ── Run inference ──────────────────────────────────────────────
    inference_start = time.time()
    result = predict(all_features, valence)
    inference_elapsed = time.time() - inference_start

    total_elapsed = time.time() - pipeline_start

    # ── Display result ─────────────────────────────────────────────
    print()
    print("=" * 60)
    print("PREDICTION")
    print("=" * 60)
    print(f"  Verdict:            {result['prediction'].upper()}")
    print(f"  Confidence:         {result['confidence'] * 100:.1f}%")
    print(f"  P(truth):           {result['raw_probabilities']['truth']:.4f}")
    print(f"  P(lie):             {result['raw_probabilities']['lie']:.4f}")
    print()
    print(f"Inference time:       {inference_elapsed * 1000:.1f} ms")
    print(f"Total pipeline time:  {total_elapsed:.1f} s")
    print()
    print(f"Transcript preview: {transcript[:150]}{'...' if len(transcript) > 150 else ''}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test_inference.py <video_path> <positive|negative>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])