import time
import cv2
from flask import Flask, Response
from flask_cors import CORS

try:
    from picamera2 import Picamera2
except ImportError:
    raise ImportError("Picamera2 library is required on Raspberry Pi 5")

# Terminal colors
GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"

# Flask app
app = Flask(__name__)
CORS(app)

# Camera setup
def setup_camera():
    picam = Picamera2()
    config = picam.create_video_configuration(
        main={"size": (1280, 720), "format": "RGB888"},
        controls={"FrameDurationLimits": (33333, 33333)}  # ~30 fps
    )
    picam.configure(config)
    picam.start()
    time.sleep(2)
    print(f"{GREEN}[INFO]{RESET} Camera initialized.")
    return picam

picam2 = setup_camera()

# FPS tracking
last_cam_time = time.time()
cam_fps = 0.0
frame_count = 0

# Frame generator
def generate_frames():
    global last_cam_time, cam_fps, frame_count
    while True:
        try:
            frame = picam2.capture_array()
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
            if not ret:
                continue

            frame_count += 1
            now = time.time()
            cam_fps = 1.0 / (now - last_cam_time + 1e-10)
            last_cam_time = now

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            print(f"{RED}[ERROR]{RESET} Camera capture failed: {e}")
            time.sleep(0.1)

# Routes
@app.route('/video')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <html>
        <head>
            <title>Raspberry Pi 5 Camera Streaming</title>
            <style>
                body {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    background-color: #111;
                    color: #eee;
                    font-family: Arial, sans-serif;
                    height: 100vh;
                    margin: 0;
                }
                h2 { margin-bottom: 20px; }
                img { border: 3px solid #444; border-radius: 10px; }
            </style>
        </head>
        <body>
            <h2>Raspberry Pi 5 Camera Streaming</h2>
            <img src='/video' width='1080'>
        </body>
    </html>
    """

# Main
if __name__ == '__main__':
    print(f"{GREEN}[INFO]{RESET} Starting Flask camera server on 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)