import cv2
import numpy as np
import tflite_runtime.interpreter as tflite


def load_labels(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


# Example usage:
label_map_path = "labels/labels-ssd.txt"
# using function parse
labels = load_labels(label_map_path)


def draw_boxes(frame, boxes, labels, classes, scores, threshold=0.6):  # noqa
    height, width, _ = frame.shape

    for i in range(len(scores)):
        if scores[i] > threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)
            object_name = labels[int(classes[i])]

            # Prepare text with COCO label and confidence score
            label_with_score = "{}: {:.2f}".format(object_name, scores[i])
            # Draw the bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            # Display text above the bounding box
            cv2.putText(
                frame,
                label_with_score,
                (xmin, ymin - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )


# Load the TFLite model and allocate tensors
# interpretation is a built in function in the library
interpreter = tflite.Interpreter(
    model_path="tflite-models/ssd_mobilenet.tflite"
)  # noqa
interpreter.allocate_tensors()

# Get input and output details to display on camera window
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Check the input shape to resize images accordingly
input_shape = input_details[0]["shape"]

# Initialize camera
cap = cv2.VideoCapture(0)

while True:
    capture, frame = cap.read()
    if not capture:
        break

    # Resize and preprocess frame
    frame_resized = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(frame_resized, axis=0).astype(np.uint8)

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]["index"], input_data)

    # Run the inference
    interpreter.invoke()

    # Retrieve the output of the model
    boxes = interpreter.get_tensor(output_details[0]["index"])[
        0
    ]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[1]["index"])[
        0
    ]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]["index"])[
        0
    ]  # Confidence of detected objects
    num = interpreter.get_tensor(output_details[3]["index"])[
        0
    ]  # Total number of detections

    # Draw the bounding boxes on the original frame
    draw_boxes(frame, boxes, labels, classes, scores)

    # Display the result
    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
