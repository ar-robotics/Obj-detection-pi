from interfaces.msg import Message, SwitchCamera
from threading import Thread

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import rclpy
from rclpy.node import Node
from flask import Flask, Response

from .utils import get_config, get_path, Camera


last_frame: bytes | None = None
app = Flask(__name__)


@app.route("/snapshot")
def snapshot():
    # frame = process_frame_for_detection()
    if last_frame is None:
        return "Failed to capture frame", 400

    return Response(last_frame, mimetype="image/jpeg")


class ObjectDetection(Node):

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
        fps = self.config["fps"]
        self.is_obj_detection_enabled = self.config["enable_detection"]

        self.current_camera = Camera.ARM
        self.cap = cv2.VideoCapture(self.current_camera.value)

        # timers
        self.create_timer(1 / fps, self.process_frame_for_detection)

        # Load COCO labels
        self.coco_labels = self.parse_label_map()

        # Load the TFLite model and allocate tensors
        self.interpreter = tflite.Interpreter(
            model_path=get_path("/tflite-models/ssd_mobilenet.tflite")
        )
        self.interpreter.allocate_tensors()

        # Get input and output detailsq
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        print("done with init")

    def _switch_camera(self, camera: Camera) -> None:
        self.cap.release()
        self.current_camera = camera
        self.cap = cv2.VideoCapture(camera.value)

    def handle_switch_camera(self, msg):
        camera = Camera(msg.camera)

        if camera == self.current_camera:
            return

        if camera == "arm":
            # switch to arm camera
            # disable object detection?
            pass

        if camera == "drive":
            # switch to drive camera
            # object detection should be on
            pass

    @staticmethod
    def parse_label_map() -> dict:
        label_map = {}
        label_map_path = get_path("/labels/mscoco_complete_label.pbtxt")

        with open(label_map_path, "r") as file:
            lines = file.readlines()
            current_id = None

            for line in lines:
                if "id:" in line:
                    current_id = int(line.strip().split(" ")[-1])
                elif "display_name:" in line:
                    display_name = line.strip().split('"')[1]
                    label_map[current_id] = display_name

        return label_map

    def draw_boxes(self, frame, boxes, classes, scores, threshold=0.5):
        height, width, _ = frame.shape

        for i in range(len(scores)):
            if scores[i] > threshold:
                ymin, xmin, ymax, xmax = boxes[i]
                xmin = int(xmin * width)
                xmax = int(xmax * width)
                ymin = int(ymin * height)
                ymax = int(ymax * height)
                class_id = int(classes[i])
                label = self.coco_labels.get(class_id, "Unknown")
                label_with_score = "{}: {:.2f}".format(label, scores[i])

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label_with_score,
                    (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

    def process_frame_for_detection(self) -> None:
        ret, frame = self.cap.read()
        if not ret:
            return None

        frame_resized = cv2.resize(
            frame,
            (self.input_details[0]["shape"][2], self.input_details[0]["shape"][1]),
        )
        input_data = np.expand_dims(frame_resized, axis=0).astype(np.uint8)

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]["index"])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]["index"])[0]

        self.draw_boxes(frame, boxes, classes, scores)

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
    print("flask server started")
    node = ObjectDetection()
    flask_thread.start()
    rclpy.spin(node)
    rclpy.shutdown()
