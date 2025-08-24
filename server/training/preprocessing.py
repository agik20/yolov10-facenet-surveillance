# preprocessing.py
import os
from PIL import Image
import torch
from facenet_pytorch import MTCNN

# === Konfigurasi Device dan MTCNN ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🚀 [INFO] Menggunakan device: {device}")

# MTCNN untuk deteksi dan resize wajah (ukuran output 160x160)
mtcnn = MTCNN(image_size=160, margin=14, device=device)

def load_faces(folder_path):
    """Muat wajah dari folder dan resize ke 160x160."""
    face_tensors = []
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.jpg', '.png')):
            continue

        img_path = os.path.join(folder_path, filename)
        try:
            img = Image.open(img_path).convert('RGB')
            face = mtcnn(img)
            if face is not None:
                face_tensors.append(face)
        except Exception as e:
            print(f"⚠️ [WARNING] Gagal memuat {filename}: {e}")

    print(f"✅ [INFO] Total wajah berhasil dimuat: {len(face_tensors)}")
    return face_tensors