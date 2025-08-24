import os
import cv2
import numpy as np
from glob import glob
import random
from tqdm import tqdm

# Konfigurasi direktori dan jumlah augmentasi
INPUT_ROOT = "data_faces/train"
OUTPUT_ROOT = "augmented_faces/train"
NUM_AUGMENTED = 100

def rotate(image, angle):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def flip(image):
    return cv2.flip(image, 1)

def adjust_brightness_contrast(image, brightness, contrast):
    return cv2.convertScaleAbs(image, alpha=1 + contrast / 100.0, beta=brightness)

def zoom(image, zoom_factor):
    h, w = image.shape[:2]
    new_w, new_h = int(w / zoom_factor), int(h / zoom_factor)
    center_x, center_y = w // 2, h // 2
    cropped = image[center_y - new_h // 2:center_y + new_h // 2,
                    center_x - new_w // 2:center_x + new_w // 2]
    return cv2.resize(cropped, (w, h))

def add_gaussian_noise(image, std):
    noise = np.random.normal(0, std, image.shape).astype(np.int16)
    noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy

def apply_blur(image, ksize=3):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def downscale_resolution(image, scale=0.5):
    h, w = image.shape[:2]
    small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def augment_image(img, num_variants):
    augmented = []
    for _ in range(num_variants):
        aug = img.copy()

        if random.random() < 1.0:
            angle = random.uniform(-15, 15)
            aug = rotate(aug, angle)

        if random.random() < 0.7:
            aug = flip(aug)

        if random.random() < 1.0:
            brightness = random.randint(-30, 30)
            contrast = random.randint(-20, 20)
            aug = adjust_brightness_contrast(aug, brightness, contrast)

        if random.random() < 1.0:
            zoom_factor = random.uniform(1.0, 1.2)
            aug = zoom(aug, zoom_factor)

        if random.random() < 0.7:
            std = random.uniform(3, 8)
            aug = add_gaussian_noise(aug, std)

        if random.random() < 0.5:
            k = random.choice([3, 5])
            aug = apply_blur(aug, k)

        if random.random() < 0.5:
            scale = random.uniform(0.4, 0.7)
            aug = downscale_resolution(aug, scale)

        augmented.append(aug)
    return augmented

def main():
    print("Memulai proses augmentasi data wajah...\n")

    person_folders = glob(os.path.join(INPUT_ROOT, "*"))
    total_person = len(person_folders)
    print(f"Ditemukan {total_person} folder individu pada direktori: {INPUT_ROOT}\n")

    for person_path in tqdm(person_folders, desc="Memproses individu", unit="orang"):
        if not os.path.isdir(person_path):
            continue

        person_name = os.path.basename(person_path)
        input_img_dir = os.path.join(INPUT_ROOT, person_name)
        output_img_dir = os.path.join(OUTPUT_ROOT, person_name)
        ensure_dir(output_img_dir)

        image_files = glob(os.path.join(input_img_dir, "*.jpg"))
        print(f"\nIndividu: {person_name} | Jumlah gambar: {len(image_files)}")

        for img_file in tqdm(image_files, desc="  Memproses gambar", leave=False):
            img = cv2.imread(img_file)
            if img is None:
                print(f"Gagal membaca gambar: {img_file}")
                continue

            base_name = os.path.splitext(os.path.basename(img_file))[0]
            cv2.imwrite(os.path.join(output_img_dir, f"{base_name}_orig.jpg"), img)

            aug_imgs = augment_image(img, num_variants=NUM_AUGMENTED)
            for i, aug_img in enumerate(aug_imgs):
                out_path = os.path.join(output_img_dir, f"{base_name}_aug{i+1}.jpg")
                cv2.imwrite(out_path, aug_img)

    print("\nProses augmentasi selesai. Semua data disimpan di:", OUTPUT_ROOT)

if __name__ == "__main__":
    main()