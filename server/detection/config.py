BASE_DIR = "/home/gotham/lomba/facenet/inference"
PI_STREAM_URL = "http://pi:5000/video"
YOLO_IMG_SIZ = 512
YOLO_CONF_THRESHOLD = 0.4
RECOG_CONF_THRESHOLD = 0.5
FRAME_FPS_CAP = 30.0
MIN_PROC_INTERVAL = 1.0 / FRAME_FPS_CAP
PI_HOST = "pi"
PI_USER = "gotham"
PI_VENV_PATH = "/home/gotham/lomba/venv/bin/activate"
PI_STREAM_SCRIPT = "/home/gotham/lomba/streaming/stream_flask.py"

# Terminal colors
GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"