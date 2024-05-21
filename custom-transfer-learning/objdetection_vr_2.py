from flask import Flask, Response
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

app = Flask(__name__)


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


# Load model and labels
model_path = "/home/pi/Obj-detection-pi/custom-transfer-learning/tflite_models/people_detection_2.tflite"
label_path = "/home/pi/Obj-detection-pi/custom-transfer-learning/labels/labels.txt"
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = load_labels(label_path)


def getCaps():
    print("Getting camera")
    return cv2.VideoCapture(-1)


cap = getCaps()


# Function to draw bounding boxes on the frame
def draw_boxes(frame, num_detections, boxes, classes, scores, labels, threshold=0.4):
    """Draw bounding boxes on the frame.
    Args:
        frame: the frame to draw on.
        num_detections: the number of detections.
        boxes: the bounding boxes.
        classes: the class of the detected object.
        scores: the confidence scores of the detected object.
        labels: the labels of the detected object.
        threshold: the confidence threshold to use.
    R
    """
    height, width, _ = frame.shape
    for i in range(num_detections):
        if scores[i] > threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            object_name = labels[int(classes[i])]
            label = "%s" % (object_name)
            labelSize, baseLine = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )  # Get font size
            ymin = max(ymin, labelSize[1])
            cv2.rectangle(
                frame,
                (int(xmin), int(ymin - round(1.5 * labelSize[1]))),
                (int(xmin + round(1.5 * labelSize[0])), int(ymin + baseLine)),
                (255, 255, 255),
                cv2.FILLED,
            )
            cv2.putText(
                frame,
                label,
                (int(xmin), int(ymin)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )


def getFrame():
    # cap = getCaps()
    ret, frame = cap.read()
    return ret, frame


# Function to process a single frame for detection and return the frame
def process_frame_for_detection(frame=None):

    ret, frame = frame  # cap.read()  # Capture frame
    # cap.release()

    if not ret:
        return None  # Return None if frame capture failed

    height = input_details[0]["shape"][1]
    width = input_details[0]["shape"][2]
    # Prepare the frame for model input

    input_frame = cv2.resize(frame, (width, height))
    input_frame = np.expand_dims(input_frame, axis=0).astype(np.uint8)

    # Run inference
    interpreter.set_tensor(input_details[0]["index"], input_frame)
    interpreter.invoke()

    num_detections = int(interpreter.get_tensor(output_details[2]["index"])[0])
    scores = interpreter.get_tensor(output_details[0]["index"])[0]
    boxes = interpreter.get_tensor(output_details[1]["index"])[0]
    classes = interpreter.get_tensor(output_details[3]["index"])[0]

    # Draw bounding boxes on the original frame
    draw_boxes(frame, num_detections, boxes, classes, scores, labels)

    # Convert the frame to JPEG format
    ret, buffer = cv2.imencode(".jpg", frame)
    if not ret:
        return None

    return buffer.tobytes()


@app.route("/snapshot")
def snapshot():

    frame = process_frame_for_detection(getFrame())
    if frame is None:
        return "Failed to capture frame", 400
    return Response(frame, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
