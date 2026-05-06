from pathlib import Path

import opensmile


_SMILE = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)


def extract_opensmile(audio_path: Path) -> dict:
    
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    result = _SMILE.process_file(str(audio_path))

    if result is None or len(result) == 0:
        raise RuntimeError(f"OpenSMILE produced no output for {audio_path}")

    # result is a DataFrame with one row and ~88 columns; flatten to dict
    row = result.iloc[0]
    features = {f"audio_{col}": float(row[col]) for col in result.columns}
    return features
