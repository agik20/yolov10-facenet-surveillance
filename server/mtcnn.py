import cv2
import time
from PIL import Image
import torch
from facenet_pytorch import MTCNN
import os

# Pilih device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)

# Input video
input_path = r"/home/gotham/lomba/face_detection/Video/input/skenario_ujian2.mp4"
cap = cv2.VideoCapture(input_path)

# Output video
output_path = r"/home/gotham/lomba/face_detection/Video/output/skenario_ujian2_result.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Ambil jumlah total frame
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_count = 0

# Variabel untuk FPS
prev_time = time.time()

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

    # Konversi BGR -> RGB -> PIL
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)

    # Deteksi wajah
    boxes, probs = mtcnn.detect(pil_img)

    # Gambar kotak
    if boxes is not None:
        for box, prob in zip(boxes, probs):
            if prob is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{prob:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Tambah FPS text
    cv2.putText(frame, f"FPS: {fps_text:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Simpan ke output video
    out.write(frame)

    # Tampilkan progress di terminal setiap 10 frame
    if frame_count % 10 == 0 or frame_count == total_frames:
        print(f"Frame {frame_count}/{total_frames} diproses...")

# Tutup semua
cap.release()
out.release()
cv2.destroyAllWindows()

print("\nProses selesai!")
print(f"Video hasil sudah disimpan di: {output_path}")
