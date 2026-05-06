"""Extract WAV audio track from a video file.

Uses moviepy (which wraps ffmpeg) to extract audio at 16kHz mono — the
format OpenSMILE eGeMAPS and Whisper both expect.
"""

from pathlib import Path
from moviepy import VideoFileClip


def extract_audio(video_path: Path, output_dir: Path) -> Path:
    """Extract audio from a video file as a 16kHz mono WAV.

    Args:
        video_path: Path to the input video (.mp4, .wmv, etc.)
        output_dir: Directory where the WAV will be written

    Returns:
        Path to the extracted WAV file.

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If extraction fails (e.g., video has no audio track).
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / f"{video_path.stem}.wav"

    try:
        clip = VideoFileClip(str(video_path))
        if clip.audio is None:
            raise RuntimeError(f"Video has no audio track: {video_path}")

        # 16kHz mono — standard for speech processing
        clip.audio.write_audiofile(
            str(audio_path),
            fps=16000,
            nbytes=2,         # 16-bit samples
            codec='pcm_s16le',
            ffmpeg_params=['-ac', '1'],  # mono
            logger=None,      # suppress moviepy's progress bars
        )
        clip.close()
    except Exception as e:
        raise RuntimeError(f"Audio extraction failed for {video_path}: {e}")

    if not audio_path.exists():
        raise RuntimeError(f"Audio extraction produced no output at {audio_path}")

    return audio_path
