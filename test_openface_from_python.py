import subprocess
from pathlib import Path
import time

video_path = Path(r"C:\openface_work\BF001_1PT.mp4")
output_dir = Path(r"C:\openface_work\python_test_output")
output_dir.mkdir(parents=True, exist_ok=True)

video_dir = video_path.parent.absolute()
video_name = video_path.name


start = time.time()
print(f'Start time: {start}')
time.sleep(1)

cmd = [
    "docker", "run", "--rm",
    "-v", f"{video_dir}:/in",
    "-v", f"{output_dir.absolute()}:/out",
    "my-openface:latest",
    "/opt/OpenFace/build/bin/FeatureExtraction",
    "-f", f"/in/{video_name}",
    "-out_dir", "/out",
]

print("Running:", " ".join(cmd))
result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
print("Return code:", result.returncode)
print("Stdout (last 500 chars):", result.stdout[-500:])
if result.returncode != 0:
    print("Stderr:", result.stderr)

end = time.time()
print(f'Elapsed: {end - start:.2f} seconds')