"""Transcribe audio to text using OpenAI Whisper.

Loads the Whisper model once at module import (slow first time, ~5s).
Subsequent transcriptions reuse the loaded model.

The 'base' model is used for balance of speed and accuracy. It transcribes
at roughly 2-3x realtime on CPU for clean speech, and preserves fillers
('uh', 'um') by default.
"""

from pathlib import Path

import whisper


WHISPER_MODEL_NAME = "base"

# Load once at import — this downloads the model on first run (~150MB)
_MODEL = whisper.load_model(WHISPER_MODEL_NAME)


def transcribe(audio_path: Path) -> str:
    """Transcribe an audio file to text.

    Args:
        audio_path: Path to a WAV file (16kHz mono recommended; matches what
                    audio.py produces)

    Returns:
        The transcript text as a single string with whitespace normalised.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        RuntimeError: If Whisper returns no text.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    result = _MODEL.transcribe(str(audio_path), fp16=False)

    transcript = result.get("text", "").strip()
    if not transcript:
        raise RuntimeError(f"Whisper produced empty transcript for {audio_path}")

    return transcript
