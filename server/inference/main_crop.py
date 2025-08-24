import os
import sys
import gc
import cv2
import time
import json
import torch
import joblib
import queue
import threading
import subprocess
import numpy as np
from collections import deque, defaultdict, Counter
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1

# Optional NMS (untuk kurangi box ganda)
try:
    import torchvision
    HAS_TORCHVISION = True
except Exception:
    HAS_TORCHVISION = False

# Terminal colors
GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"

# =========================
# Environment & device
# =========================
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{GREEN}[INFO]{RESET} Device: {'GPU' if device.type == 'cuda' else 'CPU'}")

# =========================
# Paths & config
# =========================
BASE_DIR = "/home/gotham/lomba/facenet/inference"
PI_STREAM_URL = "http://pi:5000/video"

# Detector & recognition thresholds
YOLO_IMG_SIZ = 640
YOLO_CONF_THRESHOLD = 0.6        # dinaikkan dari 0.4 untuk kurangi box ganda
NMS_IOU_THRESHOLD = 0.4          # NMS tambahan
RECOG_MIN_PROBA = 0.6            # minimal probabilitas untuk menerima nama
DECISION_MARGIN_MIN = 0.6        # fallback kalau tidak ada proba (skala relatif)

# Smoothing & runtime params
WARMUP_FRAMES = 60                # skip n frame awal
NAME_SMOOTH_WINDOW = 7           # panjang deque histori nama
NAME_SMOOTH_MIN_COUNT = 3        # minimal kemunculan nama untuk dinyatakan stabil
IOU_MATCH_THRESHOLD = 0.4        # asosiasi antar-frame untuk "track" sederhana

FRAME_FPS_CAP = 30.0
MIN_PROC_INTERVAL = 1.0 / FRAME_FPS_CAP

STABLE_TIME_REQUIRED = 3.0   # detik, minimal durasi nama stabil sebelum crop

PI_HOST, PI_USER = "pi", "gotham"
PI_VENV_PATH = "/home/gotham/lomba/venv/bin/activate"
PI_STREAM_SCRIPT = "/home/gotham/lomba/streaming/stream.py"

frame_q = queue.Queue(maxsize=1)
stop_event = threading.Event()

# User ID mapping
USER_ID_MAP = {
    "Abyan Nurfajarizqi": 1,
    "Hafidz Hidayatullah": 2,
    "Ardutra Agi Ginting": 3
}
CAM_ID = 1  # bisa diganti kalau pakai multi kamera

# Output paths
CROP_DIR = "/var/www/html/gotham/assets/output/crop_img"
FULL_DIR = "/var/www/html/gotham/assets/output/full_img"
os.makedirs(CROP_DIR, exist_ok=True)
os.makedirs(FULL_DIR, exist_ok=True)

# Interval penyimpanan (detik)
SAVE_INTERVAL = 300  # 5 menit
last_save_time = {}

# =========================
# Load models
# =========================
print(f"{GREEN}[INFO]{RESET} Loading models...")
detector = YOLO(os.path.join(BASE_DIR, "models", "yolov8n-face.pt"))
facenet = InceptionResnetV1(pretrained="vggface2").eval()
facenet = facenet.to(device).half() if device.type == "cuda" else facenet.to(device)
encoder = joblib.load(os.path.join(BASE_DIR, "models", "in_encoder.pkl"))
pca_path = os.path.join(BASE_DIR, "models", "pca_model.pkl")
pca = joblib.load(pca_path) if os.path.exists(pca_path) else None

try:
    from cuml.preprocessing import Normalizer
    normalizer = Normalizer(norm="l2")
except Exception:
    normalizer = None

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

# =========================
# Utils
# =========================
def start_pi_stream():
    remote_cmd = f"bash -c 'source {PI_VENV_PATH} && nohup python3 {PI_STREAM_SCRIPT} >/dev/null 2>&1 &'"
    print(f"{GREEN}[INFO]{RESET} Starting Pi stream...")
    try:
        subprocess.Popen(["ssh", "-f", f"{PI_USER}@{PI_HOST}", remote_cmd],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)
    except Exception as e:
        print(f"{RED}[ERROR]{RESET} Failed to start Pi stream: {e}")

def stop_pi_stream():
    try:
        subprocess.run(f"ssh {PI_USER}@{PI_HOST} pkill -f '{PI_STREAM_SCRIPT}'",
                       shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"{RED}[INFO]{RESET} Pi stream stopped.")
    except Exception as e:
        print(f"{RED}[ERROR]{RESET} Failed to stop Pi stream: {e}")

def preprocess_face(img_rgb, box, size=160, stream=None):
    x1, y1, x2, y2 = map(int, box)
    h, w = img_rgb.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    face = cv2.resize(img_rgb[y1:y2, x1:x2], (size, size))
    t = torch.from_numpy(np.transpose(face, (2, 0, 1)).astype(np.float32) / 255.0).unsqueeze(0)
    if device.type == "cuda":
        with torch.cuda.stream(stream or torch.cuda.current_stream()):
            t = t.to(device, non_blocking=True).half()
    else:
        t = t.to(device)
    return t

def iou(boxA, boxB):
    # box: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = max(0, (boxA[2]-boxA[0])) * max(0, (boxA[3]-boxA[1]))
    areaB = max(0, (boxB[2]-boxB[0])) * max(0, (boxB[3]-boxB[1]))
    union = areaA + areaB - inter + 1e-9
    return inter / union

def greedy_match_tracks(prev_boxes, prev_ids, curr_boxes, iou_thresh=0.4):
    """
    Cocokkan boxes current ke track sebelumnya berdasar IoU (greedy).
    return: list track_id untuk setiap curr_box (panjang = len(curr_boxes)),
            updated prev_boxes/ids (untuk referensi luar di-update manual).
    """
    assigned = [-1] * len(curr_boxes)
    if not prev_boxes:
        return assigned

    used_prev = set()
    for i, cb in enumerate(curr_boxes):
        best_j = -1
        best_iou = 0.0
        for j, pb in enumerate(prev_boxes):
            if j in used_prev:
                continue
            iou_val = iou(cb, pb)
            if iou_val > best_iou:
                best_iou, best_j = iou_val, j
        if best_j != -1 and best_iou >= iou_thresh:
            assigned[i] = prev_ids[best_j]
            used_prev.add(best_j)
    return assigned

# =========================
# Frame grabber
# =========================
class FrameGrabber(threading.Thread):
    def __init__(self, url, q, stop_evt):
        super().__init__(daemon=True)
        self.url, self.q, self.stop_evt = url, q, stop_evt
        self.cap, self.prev_time, self.fps = None, time.time(), 0.0
        self.frame_interval = 1.0 / FRAME_FPS_CAP  # Target time between frames

    def _open(self):
        if self.cap: self.cap.release()
        self.cap = cv2.VideoCapture(self.url)
        if self.cap.isOpened():
            print(f"{GREEN}[STATUS]{RESET} Stream opened successfully")
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            print(f"{RED}[ERROR]{RESET} Failed to open stream")

    def run(self):
        self._open()
        next_frame_time = time.time()
        while not self.stop_evt.is_set():
            if not self.cap or not self.cap.isOpened():
                time.sleep(0.5)
                self._open()
                continue

            current_time = time.time()
            if current_time < next_frame_time:
                time.sleep(max(0, next_frame_time - current_time))
            
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.1)
                continue
            
            next_frame_time += self.frame_interval
            if next_frame_time < current_time:
                next_frame_time = current_time + self.frame_interval

            ts = time.time()
            self.fps = 1.0 / (ts - self.prev_time + 1e-10)
            self.prev_time = ts
            
            try:
                self.q.put_nowait((frame, ts))
            except queue.Full:
                try:
                    self.q.get_nowait()
                    self.q.put_nowait((frame, ts))
                except queue.Empty:
                    pass
        if self.cap:
            self.cap.release()
        print(f"{RED}[INFO]{RESET} FrameGrabber stopped.")

# =========================
# Inference worker
# =========================
class InferenceWorker(threading.Thread):
    def __init__(self, q, stop_evt, grabber):
        super().__init__(daemon=True)
        self.q, self.stop_evt, self.grabber = q, stop_evt, grabber
        self.frame_count, self.rekap_data = 0, []
        self.cuda_stream = torch.cuda.Stream() if device.type == "cuda" else None
        self.prev_time, self.fps = time.time(), 0.0
        self.last_print_time, self.last_names = 0, []
        self.svm = joblib.load(os.path.join(BASE_DIR, "models", "svm_model.pkl"))

        # Tracking & smoothing state
        self.prev_boxes = []     # boxes frame sebelumnya
        self.prev_ids = []       # track_id tiap prev_box
        self.next_track_id = 1
        self.name_hist = defaultdict(lambda: deque(maxlen=NAME_SMOOTH_WINDOW))
        self.name_first_seen = {}  # track_id -> { "name": str, "time": float }

    def _predict_names(self, emb_np):
        """
        Kembalikan (names, confidences) untuk setiap embedding.
        names: list of str (prediksi nama atau "Unknown")
        confidences: list of float (confidence / margin)
        """
        names, confs = [], []
        # Normalisasi L2 (jika belum)
        if normalizer is None:
            # normalize manual
            norms = np.linalg.norm(emb_np, axis=1, keepdims=True) + 1e-10
            emb_np = emb_np / norms
        # PCA (opsional)
        if pca is not None:
            emb_np = pca.transform(emb_np)

        # Coba probabilitas
        use_proba = hasattr(self.svm, "predict_proba")
        if use_proba:
            try:
                proba = self.svm.predict_proba(emb_np)  # shape: [N, K]
            except Exception:
                proba = None
                use_proba = False
        else:
            proba = None

        if use_proba and proba is not None:
            preds = np.argmax(proba, axis=1)
            maxp = np.max(proba, axis=1)
            for i, pid in enumerate(preds):
                if maxp[i] < RECOG_MIN_PROBA:
                    names.append("Unknown")
                    confs.append(float(maxp[i]))
                else:
                    try:
                        name = encoder.inverse_transform([int(pid)])[0]
                    except Exception:
                        name = str(int(pid))
                    names.append(name)
                    confs.append(float(maxp[i]))
        else:
            # Fallback: decision_function (margin)
            # Catatan: skala margin berbeda-beda; gunakan threshold heuristik
            has_dec = hasattr(self.svm, "decision_function")
            if has_dec:
                dec = self.svm.decision_function(emb_np)  # [N, K] atau [N] untuk binary
                if dec.ndim == 1:
                    # Binary: margin positif → class 1, negatif → class 0
                    preds = (dec > 0).astype(int)
                    margin = np.abs(dec)
                    for i, pid in enumerate(preds):
                        if margin[i] < DECISION_MARGIN_MIN:
                            names.append("Unknown")
                            confs.append(float(margin[i]))
                        else:
                            try:
                                name = encoder.inverse_transform([int(pid)])[0]
                            except Exception:
                                name = str(int(pid))
                            names.append(name)
                            confs.append(float(margin[i]))
                else:
                    preds = np.argmax(dec, axis=1)
                    # gunakan gap top2 sebagai "margin"
                    sorted_dec = np.sort(dec, axis=1)
                    margin = sorted_dec[:, -1] - sorted_dec[:, -2]
                    for i, pid in enumerate(preds):
                        if margin[i] < DECISION_MARGIN_MIN:
                            names.append("Unknown")
                            confs.append(float(margin[i]))
                        else:
                            try:
                                name = encoder.inverse_transform([int(pid)])[0]
                            except Exception:
                                name = str(int(pid))
                            names.append(name)
                            confs.append(float(margin[i]))
            else:
                # Fallback paling sederhana: predict langsung
                preds = self.svm.predict(emb_np)
                for pid in preds:
                    try:
                        name = encoder.inverse_transform([int(pid)])[0]
                    except Exception:
                        name = str(int(pid))
                    names.append(name)
                    confs.append(1.0)  # tidak ada info confidence
        return names, confs

    def run(self):
        print(f"{GREEN}[INFO]{RESET} InferenceWorker started")
        try:
            while not self.stop_evt.is_set():
                try:
                    frame, _ = self.q.get(timeout=0.5)
                except queue.Empty:
                    continue

                self.frame_count += 1
                # Warm-up: skip beberapa frame awal untuk stabilitas
                if self.frame_count <= WARMUP_FRAMES:
                    # update FPS counter walau skip
                    current_time = time.time()
                    self.fps = 1.0 / (current_time - self.prev_time + 1e-10)
                    self.prev_time = current_time
                    # Cetak setiap frame warmup
                    timestamp = time.strftime("%H:%M:%S")
                    sys.stdout.write('\033[2K\r')
                    sys.stdout.write(
                        f"\r[{timestamp}] {YELLOW}[WARMUP]{RESET} Skipping frame {self.frame_count}/{WARMUP_FRAMES} ..."
                    )
                    sys.stdout.flush()
                    continue

                loop_start = time.time()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = []

                try:
                    results = detector.predict(source=rgb, imgsz=YOLO_IMG_SIZ,
                                               conf=YOLO_CONF_THRESHOLD, verbose=False)
                    if results and len(results) > 0:
                        r = results[0]
                        if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
                            boxes = r.boxes.xyxy.cpu().numpy()
                            confs = r.boxes.conf.cpu().numpy()

                            # Tambahan NMS manual (untuk berjaga-jaga dari duplikasi)
                            if HAS_TORCHVISION and len(boxes) > 1:
                                keep = torchvision.ops.nms(
                                    torch.tensor(boxes, dtype=torch.float32),
                                    torch.tensor(confs, dtype=torch.float32),
                                    iou_threshold=NMS_IOU_THRESHOLD
                                ).cpu().numpy()
                                boxes = boxes[keep]
                                confs = confs[keep]

                            # Siapkan batch untuk facenet
                            tensors, face_boxes = [], []
                            for i, box in enumerate(boxes):
                                if confs[i] < YOLO_CONF_THRESHOLD:
                                    continue
                                t = preprocess_face(rgb, box, stream=self.cuda_stream)
                                if t is not None:
                                    tensors.append(t)
                                    face_boxes.append(box)

                            if tensors:
                                batch = torch.cat(tensors, dim=0)
                                with torch.no_grad():
                                    if device.type == "cuda":
                                        batch = batch.half()
                                        with torch.cuda.stream(self.cuda_stream):
                                            emb = facenet(batch)
                                        torch.cuda.synchronize()
                                    else:
                                        emb = facenet(batch)

                                emb_np = emb.detach().cpu().numpy()
                                # Normalisasi L2 jika pakai cuml normalizer
                                if normalizer:
                                    emb_np /= (np.linalg.norm(emb_np, axis=1, keepdims=True) + 1e-10)

                                # Prediksi nama + confidence
                                pred_names, confidences = self._predict_names(emb_np)

                                # ======= Asosiasi dengan track sebelumnya (IoU) =======
                                curr_boxes_int = [[int(x) for x in fb] for fb in face_boxes]
                                assigned_ids = greedy_match_tracks(
                                    self.prev_boxes, self.prev_ids, curr_boxes_int, IOU_MATCH_THRESHOLD
                                )

                                # Assign track_id atau buat baru
                                track_ids = []
                                for idx, tid in enumerate(assigned_ids):
                                    if tid == -1:
                                        tid = self.next_track_id
                                        self.next_track_id += 1
                                    track_ids.append(tid)

                                # Update smoothing per track
                                stable_names = []
                                for idx, tid in enumerate(track_ids):
                                    raw_name = pred_names[idx]
                                    # Push ke histori track
                                    self.name_hist[tid].append(raw_name)
                                    # Majority vote
                                    most_common, count = Counter(self.name_hist[tid]).most_common(1)[0]
                                    if most_common != "Unknown" and count >= NAME_SMOOTH_MIN_COUNT:
                                        stable_name = most_common
                                    else:
                                        # Bila mayoritas masih belum stabil, tampilkan "-" agar tidak loncat-loncat
                                        stable_name = "Unknown"
                                    stable_names.append(stable_name)

                                # Simpan deteksi
                                for i in range(len(curr_boxes_int)):
                                    det = {
                                        "track_id": int(track_ids[i]),
                                        "name_raw": pred_names[i],
                                        "name": stable_names[i],
                                        "conf": float(confidences[i]),
                                        "box": [int(x) for x in curr_boxes_int[i]]
                                    }
                                    detections.append(det)

                                # Update prev boxes/ids untuk frame berikutnya
                                self.prev_boxes = curr_boxes_int
                                self.prev_ids = track_ids

                                # === Simpan gambar setiap 5 menit per user (hanya yang stabil) ===
                                for det in detections:
                                    stable_name = det["name"]
                                    tid = det["track_id"]
                                    if stable_name == "Unknown":
                                        # reset kalau sebelumnya ada
                                        if tid in self.name_first_seen:
                                            del self.name_first_seen[tid]
                                        continue

                                    now = time.time()

                                    # Reset timer jika nama berubah atau Unknown
                                    if tid not in self.name_first_seen or self.name_first_seen[tid]["name"] != stable_name:
                                        self.name_first_seen[tid] = {"name": stable_name, "time": now}
                                        continue

                                    # Jika nama sama → cek sudah berapa lama stabil
                                    first_time = self.name_first_seen[tid]["time"]
                                    elapsed_stable = now - first_time
                                    if elapsed_stable < STABLE_TIME_REQUIRED:
                                        continue  # belum cukup lama stabil → skip crop


                                    user_id = USER_ID_MAP.get(stable_name, 0)
                                    if user_id > 0:
                                        if user_id not in last_save_time or (now - last_save_time[user_id]) >= SAVE_INTERVAL:
                                            ts_str = time.strftime("%S-%M-%H")
                                            crop_filename = f"sus_{user_id}_cam_{CAM_ID}_time_{ts_str}.jpg"
                                            full_filename = f"full_sus_{user_id}_cam_{CAM_ID}_time_{ts_str}.jpg"

                                            x1, y1, x2, y2 = det["box"]
                                            face_crop = frame[y1:y2, x1:x2]

                                            cv2.imwrite(os.path.join(CROP_DIR, crop_filename), face_crop)
                                            cv2.imwrite(os.path.join(FULL_DIR, full_filename), frame)
                                            last_save_time[user_id] = now
                                            sys.stdout.write("\033[2K\r")
                                            sys.stdout.flush()
                                            print(f"{GREEN}[SAVE]{RESET} Gambar user {stable_name} disimpan.")
                except Exception as e:
                    print(f"{RED}[ERROR]{RESET} Inference: {e}")

                # =========================
                # Logging & pacing
                # =========================
                current_time = time.time()
                self.fps = 1.0 / (current_time - self.prev_time + 1e-10)
                self.prev_time = current_time
                names_display = [d["name"] for d in detections] if detections else ["-"]

                if (current_time - self.last_print_time > 0.2 or names_display != self.last_names):
                    camera_fps_str = f"{self.grabber.fps:5.1f}"
                    infer_fps_str = f"{self.fps:5.1f}"
                    faces_count_str = f"{len(detections):2d}"
                    names_str = ', '.join(names_display)[:40].ljust(40)
                    timestamp = time.strftime("%H:%M:%S")
                    
                    sys.stdout.write('\033[2K\r')
                    sys.stdout.write(
                        f"\r[{timestamp}] {YELLOW}[STATUS]{RESET} Camera: {camera_fps_str} | "
                        f"Infer: {infer_fps_str} | Faces: {faces_count_str} | Names: {names_str}"
                    )
                    sys.stdout.flush()
                    
                    self.last_print_time, self.last_names = current_time, names_display

                self.rekap_data.append({
                    "frame": self.frame_count,
                    "timestamp": time.time(),
                    "faces": detections
                })
                if len(self.rekap_data) >= 10:
                    try:
                        with open("rekap_wajah.json", "w") as f:
                            json.dump(self.rekap_data, f, indent=2)
                    except Exception as e:
                        print(f"{RED}[ERROR]{RESET} Write rekap_wajah.json: {e}")

                elapsed = time.time() - loop_start
                if elapsed < MIN_PROC_INTERVAL:
                    time.sleep(MIN_PROC_INTERVAL - elapsed)
        except Exception as e:
            print(f"{RED}[ERROR]{RESET} InferenceWorker crashed: {e}")
        finally:
            sys.stdout.write('\033[2K\r')
            sys.stdout.flush()
            self.cleanup()
            print(f"{RED}[INFO]{RESET} InferenceWorker stopped.")

    def cleanup(self):
        try:
            with open("rekap_wajah.json", "w") as f:
                json.dump(self.rekap_data, f, indent=2)
            del self.svm
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"{RED}[ERROR]{RESET} Cleanup failed: {e}")

# =========================
# Main
# =========================
def main():
    start_pi_stream()
    grabber = FrameGrabber(PI_STREAM_URL, frame_q, stop_event)
    worker = InferenceWorker(frame_q, stop_event, grabber)

    try:
        grabber.start()
        worker.start()
        print(f"{GREEN}[INFO]{RESET} Local inference running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}[INFO]{RESET} Keyboard interrupt received. Stopping local processes...")
    finally:
        stop_event.set()
        grabber.join(timeout=2)
        worker.join(timeout=2)
        del grabber, worker
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        time.sleep(5)
        stop_pi_stream()
        print(f"{GREEN}[INFO]{RESET} Program finished safely!")

if __name__ == "__main__":
    main()