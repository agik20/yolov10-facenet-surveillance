import cv2
import time
import os
import torch
import json
from ultralytics import YOLO

# Device check
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

# Path model dan input
model_path = r"/home/gotham/lomba/face_detection/model/yolov10s-face.pt"
input_path = r"/home/gotham/lomba/face_detection/Video/input/skenario_ujian2.mp4"

# Load YOLO
model = YOLO(model_path).to(device)

# Ambil nama model dan video
model_name = os.path.splitext(os.path.basename(model_path))[0]
video_name = os.path.splitext(os.path.basename(input_path))[0]

# Buat folder output otomatis
output_dir = os.path.join("/home/gotham/Face_Detection/Video/output", model_name)
os.makedirs(output_dir, exist_ok=True)

# Buat path file output
output_path = os.path.join(output_dir, f"{video_name}_result.mp4")
meta_path = os.path.join(output_dir, f"{video_name}_meta.json")

# Setup video writer
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_count = 0
prev_time = time.time()

# Kumpulkan data statistik
fps_list = []
conf_list = []

print(f"Memproses video: {os.path.basename(input_path)}")
print(f"Total frame: {total_frames}")
print("Proses dimulai...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Hitung FPS realtime
    curr_time = time.time()
    fps_text = 1.0 / (curr_time - prev_time)
    prev_time = curr_time
    fps_list.append(fps_text)

    # Jalankan YOLOv8
    results = model(frame, verbose=False, device=device)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            conf_list.append(float(conf))

    # Tambah FPS text
    cv2.putText(frame, f"FPS: {fps_text:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Simpan output
    out.write(frame)

    # Log progress
    if frame_count % 10 == 0 or frame_count == total_frames:
        print(f"Frame {frame_count}/{total_frames} diproses...")

cap.release()
out.release()
cv2.destroyAllWindows()

# Hitung statistik
avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
avg_conf = sum(conf_list) / len(conf_list) if conf_list else 0

metadata = {
    "video": os.path.basename(input_path),
    "fps": fps,
    "resolution": [width, height],
    "avg_fps": avg_fps,
    "avg_conf": avg_conf
}

# Simpan JSON metadata
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)

print("\nProses selesai!")
print("Hasil video   :", output_path)
print("Hasil metadata:", meta_path)
