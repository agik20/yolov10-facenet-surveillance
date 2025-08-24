import queue, threading, time, gc
from config import PI_STREAM_URL, PI_USER, PI_HOST, PI_VENV_PATH, PI_STREAM_SCRIPT, GREEN, RED, RESET
from grabber.pi_stream import start_pi_stream, stop_pi_stream
from grabber.frame_grabber import FrameGrabber
from inference.worker import InferenceWorker
from inference.models import load_models

frame_q = queue.Queue(maxsize=1)
stop_event = threading.Event()

def main():
    start_pi_stream(PI_USER, PI_HOST, PI_VENV_PATH, PI_STREAM_SCRIPT)
    detector, facenet, svm, encoder, pca, normalizer, device = load_models()
    grabber = FrameGrabber(PI_STREAM_URL, frame_q, stop_event)
    worker = InferenceWorker(frame_q, stop_event, grabber, detector, facenet, svm, encoder, pca, normalizer)

    grabber.start()
    worker.start()
    print(f"{GREEN}[INFO]{RESET} Local inference running. Press Ctrl+C to stop.")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print(f"\nStopping processes...")
        stop_event.set()
        grabber.join(timeout=2)
        worker.join(timeout=2)
        del grabber, worker; gc.collect()
        stop_pi_stream(PI_USER, PI_HOST, PI_STREAM_SCRIPT)
        print(f"{GREEN}[INFO]{RESET} Program finished safely!")

if __name__ == "__main__":
    main()