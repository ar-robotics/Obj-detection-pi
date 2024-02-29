import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter


def load_labels(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]["index"]
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output(interpreter, score_threshold):
    boxes = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])[
        0
    ]  # Bounding box coordinates.
    classes = interpreter.get_tensor(interpreter.get_output_details()[1]["index"])[
        0
    ]  # Class index.
    scores = interpreter.get_tensor(interpreter.get_output_details()[2]["index"])[
        0
    ]  # Confidence scores.
    count = int(
        interpreter.get_tensor(interpreter.get_output_details()[3]["index"])[0]
    )  # Number of objects detected.

    results = []
    for i in range(count):
        if scores[i] >= score_threshold:
            result = {
                "bounding_box": boxes[i],
                "class_id": classes[i],
                "score": scores[i],
            }
            results.append(result)
    return results


def draw_results(frame, results, labels):
    for obj in results:
        ymin, xmin, ymax, xmax = obj["bounding_box"]
        xmin = int(xmin * frame.shape[1])
        xmax = int(xmax * frame.shape[1])
        ymin = int(ymin * frame.shape[0])
        ymax = int(ymax * frame.shape[0])

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(
            frame,
            labels[int(obj["class_id"])],
            (xmin, ymin - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )


# Load TFLite model and allocate tensors.
interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
output_details = interpreter.get_output_details()
print(output_details)
# Load labels.
labels = load_labels("labels.txt")

# Initialize video stream.
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to match model input.
    frame_resized = cv2.resize(frame, (160, 160))
    set_input_tensor(interpreter, frame_resized)

    interpreter.invoke()

    # Get detection results.
    results = get_output(interpreter, score_threshold=0.5)
    draw_results(frame, results, labels)

    # Display the resulting frame.
    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
