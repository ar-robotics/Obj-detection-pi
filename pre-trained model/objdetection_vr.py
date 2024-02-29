from flask import Flask, Response
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

app = Flask(__name__)


def parse_label_map(label_map_path):
    label_map = {}
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


# Load COCO labels
label_map_path = "labels/mscoco_complete_label.pbtxt"
coco_labels = parse_label_map(label_map_path)

# Load the TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path="tflite-models/ssd_mobilenet.tflite")
interpreter.allocate_tensors()

# Get input and output detailsq
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize video capture with the first webcam device
cap = cv2.VideoCapture(0)


def draw_boxes(frame, boxes, classes, scores, threshold=0.5):
    height, width, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] > threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)
            class_id = int(classes[i])
            label = coco_labels.get(class_id, "Unknown")
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


def process_frame_for_detection():
    ret, frame = cap.read()
    if not ret:
        return None

    frame_resized = cv2.resize(
        frame, (input_details[0]["shape"][2], input_details[0]["shape"][1])
    )
    input_data = np.expand_dims(frame_resized, axis=0).astype(np.uint8)

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]["index"])[0]
    classes = interpreter.get_tensor(output_details[1]["index"])[0]
    scores = interpreter.get_tensor(output_details[2]["index"])[0]

    draw_boxes(frame, boxes, classes, scores)

    ret, buffer = cv2.imencode(".jpg", frame)
    if not ret:
        return None
    return buffer.tobytes()


@app.route("/snapshot")
def snapshot():
    frame = process_frame_for_detection()
    if frame is None:
        return "Failed to capture frame", 400
    return Response(frame, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
