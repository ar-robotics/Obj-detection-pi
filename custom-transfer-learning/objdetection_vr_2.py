from threading import Thread

import cv2

from flask import Flask, Response

import numpy as np

import tflite_runtime.interpreter as tflite


# from PIL import Image

app = Flask(__name__)
# Load model and labels
model_path = "/home/pi/Obj-detection-pi/custom-transfer-learning/tflite_models/people_detection_2.tflite"
label_path = "/home/pi/Obj-detection-pi/custom-transfer-learning/labels/labels.txt"
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Function to load labels from the labels file
def load_labels(label_path):
    """Loads the labels file. Supports files with or without index numbers.

    Args:
        label_path: path to the labels file.

    Returns:
        A list with the labels.

    If the file contains index numbers, then the index number is removed from the label.
    """
    with open(label_path, "r") as file:
        labels = [line.strip() for line in file.readlines()]
    return labels


labels = load_labels(label_path)


cap = cv2.VideoCapture(-1)


class AIBox:

    def __init__(self):
        self.ret = None
        self.frame = None
        self.num_detections = 0
        self.scores = {}
        self.threshold = 0.4
        self.boxes = {}
        self.classes = {}
        self.height = 480
        self.width = 640

    def set_frame(self, ret, frame):
        self.ret = ret
        self.frame = frame

    def get_drawn_frame(self) -> bytes:
        """Draw bounding boxes on the frame.

        Args:
            frame: the frame to draw on.
            num_detections: the number of detections.
            boxes: the bounding boxes.
            classes: the class of the detected object.
            scores: the confidence scores of the detected object.
            labels: the labels of the detected object.
            threshold: the confidence threshold to use.
        """

        for i in range(self.num_detections):
            if self.scores[i] < self.threshold:
                continue

            y_min, x_min, y_max, x_max = self.boxes[i]
            x_min = int(x_min * self.width)
            x_max = int(x_max * self.width)
            y_min = int(y_min * self.height)
            y_max = int(y_max * self.height)
            cv2.rectangle(self.frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            object_name = labels[int(self.classes[i])]
            label = "%s" % (object_name)
            label_size, base_line = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )  # Get font size
            y_min = max(y_min, label_size[1])
            cv2.rectangle(
                self.frame,
                (int(x_min), int(y_min - round(1.5 * label_size[1]))),
                (int(x_min + round(1.5 * label_size[0])), int(y_min + base_line)),
                (255, 255, 255),
                cv2.FILLED,
            )
            cv2.putText(
                self.frame,
                label,
                (int(x_min), int(y_min)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode(".jpg", self.frame)

        if not ret:
            return None

        return buffer.tobytes()
        # do calculation

    def detect_in_frame(self):
        height = input_details[0]["shape"][1]
        width = input_details[0]["shape"][2]
        # Prepare the frame for model input

        input_frame = cv2.resize(self.frame, (width, height))
        input_frame = np.expand_dims(input_frame, axis=0).astype(np.uint8)

        # Run inference
        interpreter.set_tensor(input_details[0]["index"], input_frame)
        interpreter.invoke()

        self.num_detections = int(interpreter.get_tensor(output_details[2]["index"])[0])
        self.scores = interpreter.get_tensor(output_details[0]["index"])[0]
        self.boxes = interpreter.get_tensor(output_details[1]["index"])[0]
        self.classes = interpreter.get_tensor(output_details[3]["index"])[0]

    def analyze(self):
        while True:
            if self.frame is None or self.ret is None:
                continue

            self.detect_in_frame()


ai_box = AIBox()
thread = Thread(target=ai_box.analyze, daemon=True)
thread.start()


def get_frame() -> tuple:
    ret, frame = cap.read()
    return ret, frame


@app.route("/snapshot")
def snapshot():
    ret, frame = get_frame()
    ai_box.set_frame(ret, frame)

    if frame is None:
        return "Failed to capture frame", 400
    return Response(ai_box.get_drawn_frame(), mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
