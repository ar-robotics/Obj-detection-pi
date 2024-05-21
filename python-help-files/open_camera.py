import cv2

cap = cv2.VideoCapture(0)
# 0 is  the default camera

while True:
    capture, frame = (
        cap.read()
    )  # Read the frame, capture is a boolean if fram was captured or not
    if capture:
        width = frame.shape[1]
        height = frame.shape[0]
        cv2.circle(frame, (width // 2, height // 2), 2, (0, 0, 255), -1)
        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # Exit on pressing 'q'
            break
    else:
        break

cap.release()  # releases the camera
cv2.destroyAllWindows()
# he


from interfaces.msg import Message, SwitchCamera
from threading import Thread
import logging

from flask import Flask, Response
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import rclpy
from rclpy.node import Node

from .utils import get_config, get_path, Camera, visualize
from .database import Database


app = Flask(__name__)
COUNTER, FPS = 0, 0
START_TIME = time.time()


log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

last_frame: bytes | None = None
app = Flask(__name__)


@app.route("/snapshot")
def snapshot():
    if last_frame is None:
        return "Failed to capture frame", 400

    return Response(last_frame, mimetype="image/jpeg")


class ObjectDetection(Node):
    """ObjectDetection class for the object detection application"""

    def __init__(self):
        super().__init__("ObjectDetection")

        print(self.__class__.__name__, "is running!")

        # publishers
        self.pub_message = self.create_publisher(Message, "message", 1)

        # subscribers
        self.sub_switch_camera = self.create_subscription(
            SwitchCamera, "switch_camera", self.handle_switch_camera, 1
        )

        self.config = get_config()
        self.detection_enabled = self.config["enable_detection"]

        # camera
        self.current_camera = Camera.DRIVE
        self.cap = cv2.VideoCapture(self.current_camera.value)

        # frame
        self.frame_width = None
        self.frame_height = None

        self.camera_available = True
        self.flip_image = False
        self.draw_dot = False

        # timers
        self.thread = Thread(
            target=self.process_frames_for_detection, daemon=True  # noqa
        )
        self.thread.start()
        # display variables
        self.row_size = 50  # pixels
        self.left_margin = 24  # pixels
        self.text_color = (237, 237, 237)  # kromwhite
        self.font_size = 1
        self.font_thickness = 1
        self.fps_avg_frame_count = 10
        # detection model
        self.model_path = get_path("tflite-models/bolt-detection-mp.tflite")
        self.detection_frame = None
        self.detection_result_list = []
        self.detector = None
        # database
        self.db = Database()
        self.dict_info = None
        self.db_text_location = None

        self.inference()
        print("done with init")

    def _switch_camera(self, camera: Camera) -> None:
        """Switches the camera to the specified camera.

        Args:
            camera: Camera to switch to
        """
        self.camera_available = False

        self.cap.release()
        cv2.destroyAllWindows()
        self.current_camera = camera
        self.cap = cv2.VideoCapture(camera.value)
        self.camera_available = True

    def handle_switch_camera(self, msg) -> None:
        """Handles switch camera message.

        Args:
            msg: SwitchCamera message
        """
        camera = Camera[msg.camera.upper()]

        self.flip_image = msg.flip_image
        self.draw_dot = msg.draw_dot

        print(f"got switch camera message {msg} {camera}")

        if camera == self.current_camera:
            return

        camera_name = camera.name.lower()

        if camera_name == "arm":
            self._switch_camera(Camera.ARM)

        if camera_name == "drive":
            self._switch_camera(Camera.DRIVE)

    def save_result(
        self,
        result: vision.ObjectDetectorResult,
        unused_output_image: mp.Image,
        timestamp_ms: int,
    ):
        """Save the detection result to the global variable

        Args:
            result (vision.ObjectDetectorResult): The detection result
            unused_output_image (mp.Image): The output image
            timestamp_ms (int): The timestamp in milliseconds
        """
        global FPS, COUNTER, START_TIME

        # Calculate the FPS
        if COUNTER % self.fps_avg_frame_count == 0:
            FPS = self.fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()

        self.detection_result_list.append(result)
        COUNTER += 1

    def inference(self, max_results=5, score_threshold=0.8):
        """Perform the object detection inference

        Args:
            max_results (int): The maximum number of results
            score_threshold (float): The score threshold
        """
        # Create the base options for detector
        base_options = python.BaseOptions(model_asset_path=self.model_path)

        # Create the object detector options
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            max_results=max_results,
            score_threshold=score_threshold,
            result_callback=self.save_result,
        )
        # Create the object detector
        self.detector = vision.ObjectDetector.create_from_options(options)

    def _detect_objects(self, frame, detector):
        """Detect objects in the frame

        Args:
            frame (np.ndarray): The input frame
            detector (vision.ObjectDetector): The object detector

        Returns:
            np.ndarray: The frame with the detected objects
        """
        # Flip the frame horizontally
        # frame = cv2.flip(frame, 1)
        # Convert the BGR image to RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Create a MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        # Perform the object detection asynchronously
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)
        fps_text = "FPS = {:.1f}".format(FPS)
        text_location = (self.left_margin, self.row_size)
        cv2.putText(
            frame,
            fps_text,
            text_location,
            cv2.FONT_HERSHEY_DUPLEX,
            self.font_size,
            self.text_color,
            self.font_thickness,
            cv2.LINE_AA,
        )

        # Visualize results in frame
        if self.detection_result_list:
            frame, category, self.db_text_location = visualize(
                frame, self.detection_result_list[0]
            )  # noqa
            self.dict_info = self.__retreive_info(category)
            self.detection_result_list.clear()

        if self.dict_info:
            offset = self.db_text_location[1] - 35
            for key, value in self.dict_info.items():
                info_text = "{}: {}".format(key, value)
                cv2.putText(
                    frame,
                    info_text,
                    (self.db_text_location[0], offset),
                    cv2.FONT_HERSHEY_DUPLEX,
                    self.font_size - 0.3,
                    self.text_color,
                    self.font_thickness,
                    cv2.LINE_AA,
                )
                offset -= int(self.font_size * 25)

        return frame

    def process_frames_for_detection(self) -> None:
        """Process the frames for object detection."""
        while True:
            if not self.camera_available:
                continue

            self.generate_frames()

    def __retreive_info(self, object_name: str) -> dict:
        """
        Retrieve all information for the object from the database

        Args:
            object_name: Name of the object

        Returns:
            all_info_dict: Dictionary containing all information for the object
        """
        if object_name is None:
            return {}

        query = {"class": object_name.lower()}
        items_info = self.db.items.find_one(query)

        if not items_info:
            return {}

        all_info_dict = {}

        for key, value in items_info.items():
            # Save each key-value pair into the dictionary
            if key not in ["_id", "class"]:
                all_info_dict[key] = value

        return all_info_dict

    def generate_frames(self):
        """Generate the frames for the object detection's snapshot."""
        try:
            ret, frame = self.cap.read()
        except Exception as e:
            print(e)
            return

        if not ret:
            return None

        if self.flip_image:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        if self.detection_enabled:
            frame = self._detect_objects(frame, self.detector)

        if self.draw_dot:
            if self.frame_height is None or self.frame_width is None:
                self.frame_height = frame.shape[0]
                self.frame_width = frame.shape[1]

            end_effector_point = (self.frame_width / 2, self.frame_height / 2)
            frame = cv2.circle(frame, end_effector_point, 2, (0, 0, 255), -1)

        ret, buffer = cv2.imencode(".jpg", frame)

        if not ret:
            return None

        global last_frame
        last_frame = buffer.tobytes()


def main(args=None):
    rclpy.init(args=args)
    flask_thread = Thread(
        target=app.run, kwargs={"host": "0.0.0.0", "port": 5000}, daemon=True
    )
    node = ObjectDetection()
    flask_thread.start()
    print("flask server started")
    rclpy.spin(node)
    rclpy.shutdown()
import json
import enum

import numpy as np
import cv2

MARGIN = 10  # pixels
ROW_SIZE = 30  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 255, 255)


path = "/robot/src/ai_detection/ai_detection/{}"


class Camera(enum.Enum):
    DRIVE = (
        "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._Astra_Pro_HD_Camera-video-index0"
    )
    ARM = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB_2.0_Camera-video-index0"


def get_path(subpath: str) -> str:
    """Get the path of the file.

    Args:
        subpath: Subpath of the file

    Returns:
        Path of the file
    """
    return path.format(subpath)


def get_config() -> dict:
    """Get the configuration.

    Returns:
        config dict
    """
    config = {}
    path = get_path("config.json")

    with open(path, "r") as f:
        config = json.load(f)

    return config


def visualize(image, detection_result) -> np.ndarray:
    """Draws bounding boxes on the input image and return it.
    Args:
      image: The input RGB image.
      detection_result: The list of all "Detection" entities to be visualized.
    Returns:
      Image with bounding boxes.
      category_name: The category name of the detected object.
    """
    category_name = None
    text_location = None

    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        # Use the orange color for high visibility.
        cv2.rectangle(image, start_point, end_point, (46, 26, 119), 3)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2) * 100
        if probability == 100:
            probability = 99.99
        result_text = category_name + " (" + str(probability) + "%" + ")"
        text_location = (
            MARGIN + bbox.origin_x,
            MARGIN + ROW_SIZE + bbox.origin_y,
        )
        cv2.putText(
            image,
            result_text,
            text_location,
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return image, category_name, text_location
