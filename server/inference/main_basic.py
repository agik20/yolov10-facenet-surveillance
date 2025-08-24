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
import numpy as np
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1

# Terminal Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# Configuration
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{GREEN}[INFO]{RESET} Device: {'GPU' if device.type == 'cuda' else 'CPU'}")

BASE_DIR = "/home/gotham/lomba/facenet/inference"
PI_STREAM_URL = "http://pi:5000/video"
YOLO_IMG_SIZ = 512
YOLO_CONF_THRESHOLD = 0.4
RECOG_CONF_THRESHOLD = 0.5
FRAME_FPS_CAP = 30.0
MIN_PROC_INTERVAL = 1.0 / FRAME_FPS_CAP

frame_q = queue.Queue(maxsize=1)
stop_event = threading.Event()

# Load Models
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

# Preprocess Face
def preprocess_face(img_rgb, box, size=160, stream=None):
    x1, y1, x2, y2 = map(int, box)
    h, w = img_rgb.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    face = cv2.resize(img_rgb[y1:y2, x1:x2], (size, size))
    t = torch.from_numpy(
        np.transpose(face, (2, 0, 1)).astype(np.float32) / 255.0
    ).unsqueeze(0)

    if device.type == "cuda":
        with torch.cuda.stream(stream or torch.cuda.current_stream()):
            t = t.to(device, non_blocking=True).half()
    else:
        t = t.to(device)
    return t

# Frame Grabber Thread
class FrameGrabber(threading.Thread):
    def __init__(self, url, q, stop_evt):
        super().__init__(daemon=True)
        self.url, self.q, self.stop_evt = url, q, stop_evt
        self.cap = None
        self.prev_time = time.time()
        self.fps = 0.0

    def _open(self):
        self.cap = cv2.VideoCapture(self.url)
        print(f"{GREEN}[STATUS]{RESET} Stream {'opened' if self.cap.isOpened() else 'failed'}")

    def run(self):
        self._open()
        while not self.stop_evt.is_set():
            if not self.cap or not self.cap.isOpened():
                time.sleep(0.5)
                self._open()
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.1)
                continue

            ts = time.time()
            self.fps = 1.0 / (ts - self.prev_time + 1e-10)
            self.prev_time = ts

            try:
                self.q.put((frame, ts), timeout=0.01)
            except queue.Full:
                try: _ = self.q.get_nowait()
                except queue.Empty: pass
                try: self.q.put((frame, ts), timeout=0.01)
                except: pass

        if self.cap:
            self.cap.release()
        print(f"{RED}[INFO]{RESET} FrameGrabber stopped.")

# Inference Worker Thread
class InferenceWorker(threading.Thread):
    def __init__(self, q, stop_evt, grabber):
        super().__init__(daemon=True)
        self.q, self.stop_evt, self.grabber = q, stop_evt, grabber
        self.frame_count, self.rekap_data = 0, []
        self.cuda_stream = torch.cuda.Stream() if device.type == "cuda" else None
        self.prev_time, self.fps = time.time(), 0.0
        self.last_print_time = 0
        self.last_names = []

    def run(self):
        print(f"{GREEN}[INFO]{RESET} InferenceWorker started")
        svm = joblib.load(os.path.join(BASE_DIR, "models", "svm_model.pkl"))

        try:
            while not self.stop_evt.is_set():
                try:
                    frame, _ = self.q.get(timeout=0.5)
                except queue.Empty:
                    continue

                loop_start = time.time()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = []

                try:
                    results = detector.predict(source=rgb, imgsz=YOLO_IMG_SIZ, conf=YOLO_CONF_THRESHOLD, verbose=False)
                    if results and len(results) > 0:
                        r = results[0]
                        if hasattr(r, "boxes") and r.boxes:
                            boxes = r.boxes.xyxy.cpu().numpy()
                            confs = r.boxes.conf.cpu().numpy()
                            tensors, face_boxes = [], []
                            for i, box in enumerate(boxes):
                                if confs[i] < YOLO_CONF_THRESHOLD: continue
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
                                if normalizer:
                                    emb_np /= (np.linalg.norm(emb_np, axis=1, keepdims=True) + 1e-10)
                                if pca:
                                    emb_np = pca.transform(emb_np)

                                try:
                                    preds = svm.predict(emb_np)
                                except:
                                    preds = [int(svm.predict(e.reshape(1, -1))[0]) for e in emb_np]

                                for i, pred in enumerate(preds):
                                    pid = int(pred)
                                    try:
                                        name = encoder.inverse_transform([pid])[0]
                                    except:
                                        name = str(pid)
                                    if 1.0 < RECOG_CONF_THRESHOLD:
                                        name = "Unknown"
                                    detections.append({"id": pid, "name": name, "box": [int(x) for x in face_boxes[i]]})
                except Exception as e:
                    print(f"{RED}[ERROR]{RESET} Inference: {e}")

                # Update FPS & status
                current_time = time.time()
                self.frame_count += 1
                self.fps = 1.0 / (current_time - self.prev_time + 1e-10)
                self.prev_time = current_time
                names = [d["name"] for d in detections] if detections else ["-"]

                if (current_time - self.last_print_time > 0.2 or names != self.last_names):
                    camera_fps_str = f"{self.grabber.fps:5.1f}"
                    infer_fps_str = f"{self.fps:5.1f}"
                    faces_count_str = f"{len(detections):2d}"
                    names_str = ', '.join(names)
                    if len(names_str) > 40: names_str = names_str[:37] + "..."
                    sys.stdout.write('\r' + ' ' * 150 + '\r')
                    sys.stdout.write(f"{YELLOW}[STATUS]{RESET} Camera FPS: {camera_fps_str} | Infer FPS: {infer_fps_str} | Faces: {faces_count_str} | Names: {names_str}")
                    sys.stdout.flush()
                    self.last_print_time = current_time
                    self.last_names = names

                self.rekap_data.append({"frame": self.frame_count, "timestamp": time.time(), "faces": detections})
                if len(self.rekap_data) >= 10:
                    with open("rekap_wajah.json", "w") as f:
                        json.dump(self.rekap_data, f, indent=2)

                elapsed = time.time() - loop_start
                if elapsed < MIN_PROC_INTERVAL:
                    time.sleep(MIN_PROC_INTERVAL - elapsed)
        finally:
            sys.stdout.write('\r' + ' ' * 150 + '\r')
            sys.stdout.flush()
            del svm
            gc.collect()
            with open("rekap_wajah.json", "w") as f:
                json.dump(self.rekap_data, f, indent=2)
            print(f"{RED}[INFO]{RESET} InferenceWorker stopped.")

# Main
def main():
    grabber = FrameGrabber(PI_STREAM_URL, frame_q, stop_event)
    worker = InferenceWorker(frame_q, stop_event, grabber)

    try:
        grabber.start()
        worker.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sys.stdout.write('\r' + ' ' * 150 + '\r')
        sys.stdout.flush()
        print(f"{YELLOW}[INFO]{RESET} Keyboard interrupt received. Shutting down...")
    finally:
        stop_event.set()
        grabber.join(timeout=2)
        worker.join(timeout=2)
        global facenet, detector
        del facenet, detector
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"{GREEN}[INFO]{RESET} Shutdown complete. Goodbye!")

if __name__ == "__main__":
    main()