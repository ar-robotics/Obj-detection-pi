from flask import Flask, Response
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils import visualize

app = Flask(__name__)
COUNTER, FPS = 0, 0
START_TIME = time.time()
# Initialize global variables for the object detection
model_path = "/home/pi/Obj-detection-pi/custom-transfer-learning/tflite_models/bolt-detection-mp.tflite"


row_size = 50  # pixels
left_margin = 24  # pixels
text_color = (0, 0, 0)  # black
font_size = 1
font_thickness = 1
fps_avg_frame_count = 10

detection_frame = None
detection_result_list = []


def save_result(
    result: vision.ObjectDetectorResult,
    unused_output_image: mp.Image,
    timestamp_ms: int,
):
    global FPS, COUNTER, START_TIME

    # Calculate the FPS
    if COUNTER % fps_avg_frame_count == 0:
        FPS = fps_avg_frame_count / (time.time() - START_TIME)
        START_TIME = time.time()

    detection_result_list.append(result)
    COUNTER += 1


base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    max_results=5,
    score_threshold=0.70,
    result_callback=save_result,
)

detector = vision.ObjectDetector.create_from_options(options)

cap = cv2.VideoCapture("/dev/video0")


def gen_frames():

    ret, frame = cap.read()
    if not ret:
        return b""

    frame = cv2.flip(frame, 1)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    detector.detect_async(mp_image, time.time_ns() // 1_000_000)
    # Visualize results in your frame...
    # Example: frame = visualize(frame, results)
    fps_text = "FPS = {:.1f}".format(FPS)
    text_location = (left_margin, row_size)
    cv2.putText(
        frame,
        fps_text,
        text_location,
        cv2.FONT_HERSHEY_DUPLEX,
        font_size,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )

    if detection_result_list:
        frame = visualize(frame, detection_result_list[0])
        detection_result_list.clear()
    detection_frame = frame
    ret, buffer = cv2.imencode(".jpg", detection_frame)
    if not ret:
        # If frame is not encoded successfully, return an empty frame
        return b""
    # return buffer.tobytes()

    yield (
        b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + detection_frame + b"\r\n"
    )


@app.route("/video_feed")
def video_feed():
    # Change camera_id based on your configuration
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/snapshot")
def snapshot():
    frame = gen_frames()
    return Response(frame, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
