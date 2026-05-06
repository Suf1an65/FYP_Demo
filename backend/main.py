import os
import shutil
import uuid
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from feature_extraction.audio import extract_audio
from feature_extraction.head_motion import compute_head_motion
from feature_extraction.inference import predict
from feature_extraction.linguistic import extract_linguistic
from feature_extraction.openface import aggregate_openface, run_openface_docker
from feature_extraction.opensmile_wrapper import extract_opensmile
from feature_extraction.transcript import transcribe



app = FastAPI(title="Deception Detection Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BACKEND_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = BACKEND_ROOT / "temp_videos"
EXTRACTION_DIR = BACKEND_ROOT / "temp_extraction"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
EXTRACTION_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def read_root():
    return {"status": "Backend is running!"}


@app.post("/analyze-deception")
async def analyze_video(
    file: UploadFile = File(...),
    context: str = Form(...),
):
    if not file.filename or not file.filename.lower().endswith('.mp4'):
        raise HTTPException(
            status_code=400,
            detail="Only MP4 files are supported.",
        )

    if context not in ('positive', 'negative'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid context '{context}'. Must be 'positive' or 'negative'.",
        )

    request_id = uuid.uuid4().hex[:12]
    request_dir = EXTRACTION_DIR / request_id
    request_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded video
    video_path = UPLOAD_DIR / f"{request_id}_{file.filename}"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"[{request_id}] Received: {file.filename}, valence={context}")

    try:
        print(f"[{request_id}] Running OpenFace...")
        openface_csv = run_openface_docker(video_path, request_dir / "openface")
        visual = aggregate_openface(openface_csv)
        head_motion = compute_head_motion(openface_csv)

        print(f"[{request_id}] Extracting audio...")
        audio_path = extract_audio(video_path, request_dir / "audio")

        print(f"[{request_id}] Running OpenSMILE...")
        audio_features = extract_opensmile(audio_path)

        print(f"[{request_id}] Transcribing...")
        transcript = transcribe(audio_path)

        print(f"[{request_id}] Linguistic features...")
        linguistic = extract_linguistic(transcript)

        all_features = {
            **visual,
            **head_motion,
            **audio_features,
            **linguistic,
        }

        print(f"[{request_id}] Extracted {len(all_features)} features. Predicting...")

        result = predict(all_features, valence=context)

        print(
            f"[{request_id}] Prediction: {result['prediction']} "
            f"({result['confidence'] * 100:.1f}%)"
        )

        return {
            "prediction": result["prediction"].capitalize(),  
            "confidence": result["confidence"],
            "probabilities": result["raw_probabilities"],
            "transcript": transcript,
            "feature_count": len(all_features),
            "message": "Pipeline executed successfully",
        }

    except Exception as e:
        print(f"[{request_id}] ERROR: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {e}",
        )

    finally:
        if video_path.exists():
            try:
                video_path.unlink()
            except Exception:
                pass

        if request_dir.exists():
            try:
                shutil.rmtree(request_dir, ignore_errors=True)
            except Exception:
                pass