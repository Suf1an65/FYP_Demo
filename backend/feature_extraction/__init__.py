"""Feature extraction pipeline for deception detection backend.

Modules:
    openface: Run OpenFace via Docker, aggregate per-frame outputs
    opensmile_wrapper: Extract eGeMAPS audio features
    audio: Extract WAV from video
    head_motion: Compute head motion features from OpenFace raw output
"""
