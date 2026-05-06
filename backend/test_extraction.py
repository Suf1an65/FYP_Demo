"""Full-pipeline smoke test for the feature extraction modules.

Run from the backend directory:
    python test_extraction.py C:\\Users\\masuf\\Downloads\\BF001_1PT.mp4

Runs OpenFace + audio + OpenSMILE + Whisper + linguistic on a single video
and prints feature counts and sample values. Use it to verify the
extraction produces 188 features matching the training schema.
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


def main(video_path_str: str):
    video_path = Path(video_path_str)
    if not video_path.exists():
        print(f"ERROR: Video not found at {video_path}")
        sys.exit(1)

    work_dir = Path("./test_extraction_output")
    work_dir.mkdir(exist_ok=True)

    print(f"Testing extraction on: {video_path}")
    print(f"Working directory:      {work_dir.absolute()}")
    print()

    pipeline_start = time.time()

    # ── OpenFace ───────────────────────────────────────────────────
    print("Running OpenFace via Docker...")
    t0 = time.time()
    csv_path = run_openface_docker(video_path, work_dir / "openface")
    print(f"  ✓ OpenFace done in {time.time() - t0:.1f}s")
    print(f"    CSV: {csv_path}")

    openface_features = aggregate_openface(csv_path)
    print(f"  ✓ Aggregated: {len(openface_features)} features")

    head_motion_features = compute_head_motion(csv_path)
    print(f"  ✓ Head motion: {len(head_motion_features)} features")
    print()

    # ── Audio ──────────────────────────────────────────────────────
    print("Extracting audio...")
    t0 = time.time()
    audio_path = extract_audio(video_path, work_dir / "audio")
    print(f"  ✓ Audio extracted in {time.time() - t0:.1f}s")
    print(f"    WAV: {audio_path}")

    # ── OpenSMILE ──────────────────────────────────────────────────
    print("Running OpenSMILE...")
    t0 = time.time()
    opensmile_features = extract_opensmile(audio_path)
    print(f"  ✓ OpenSMILE done in {time.time() - t0:.1f}s")
    print(f"    {len(opensmile_features)} features extracted")
    print()

    # ── Transcript ─────────────────────────────────────────────────
    print("Transcribing with Whisper...")
    t0 = time.time()
    transcript = transcribe(audio_path)
    print(f"  ✓ Transcription done in {time.time() - t0:.1f}s")
    print(f"    Transcript preview: {transcript[:200]}{'...' if len(transcript) > 200 else ''}")
    print()

    # ── Linguistic ─────────────────────────────────────────────────
    print("Extracting linguistic features...")
    t0 = time.time()
    linguistic_features = extract_linguistic(transcript)
    print(f"  ✓ Linguistic done in {time.time() - t0:.1f}s")
    print(f"    {len(linguistic_features)} features extracted")
    print()

    # ── Summary ────────────────────────────────────────────────────
    all_features = {
        **openface_features,
        **head_motion_features,
        **opensmile_features,
        **linguistic_features,
    }
    pipeline_elapsed = time.time() - pipeline_start

    print(f"Total features: {len(all_features)}")
    print(f"  OpenFace visual:  {len(openface_features)}")
    print(f"  Head motion:      {len(head_motion_features)}")
    print(f"  OpenSMILE audio:  {len(opensmile_features)}")
    print(f"  Linguistic:       {len(linguistic_features)}")
    print(f"\nTotal pipeline time: {pipeline_elapsed:.1f}s")
    print()

    print("Sample features (sanity check):")
    sample_keys = [
        "AU12_r_mean",
        "AU06_r_mean",
        "pose_Rz_std",
        "pose_Ry_total_motion",
        "audio_F0semitoneFrom27.5Hz_sma3nz_amean",
        "audio_loudness_sma3_amean",
        "text_word_count",
        "text_filler_rate",
        "text_first_person_singular_rate",
        "text_third_person_rate",
        "text_sentiment_positive",
        "text_sentiment_negative",
    ]
    for k in sample_keys:
        if k in all_features:
            print(f"  {k:60s} = {all_features[k]:.4f}")
        else:
            print(f"  {k:60s} = MISSING")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_extraction.py <path_to_video>")
        sys.exit(1)
    main(sys.argv[1])