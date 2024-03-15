import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from pymongo import MongoClient


# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["mongodbVSCodePlaygroundDB"]


# # function for reading a label map, returns a dictionary with id and label name
# def parse_label_map(label_map_path):
#     label_map = {}
#     with open(label_map_path, "r") as file:
#         lines = file.read().split("\n")
#         for line in lines:
#             if "id:" in line:
#                 current_id = int(line.split("id: ")[1])
#             elif "display_name:" in line:
#                 display_name = line.split('"')[1]
#                 label_map[current_id] = display_name
#     return label_map
def load_labels(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


# Example usage:
label_map_path = "labels/labels-ssd.txt"
# using function parse
labels = load_labels(label_map_path)

employment = db["classes"]


def draw_boxes(
    frame, boxes, labels, classes, employment, scores, threshold=0.6  # noqa
):
    # Frame is the image on which we'll draw the boxes
    # Boxes are normalized to [0,1], so scale them to the frame dimensions
    height, width, _ = frame.shape

    for i in range(len(scores)):
        if scores[i] > threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)
            object_name = labels[int(classes[i])]

            query = {"type": object_name}
            items_info = employment.find_one(query)
            all_info_dict = {}
            for key, value in items_info.items():
                # Save each key-value pair into the dictionary
                if key != "_id" and key != "type":
                    all_info_dict[key] = value
            # Prepare text with COCO label and confidence score
            label_with_score = "{}: {:.2f}".format(object_name, scores[i])
            y_offset = ymin - 25
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
            for key, value in all_info_dict.items():
                # Generate the text for this key-value pair
                info_text = "{}: {}".format(key, value)
                # Display this key-value pair on the frame
                cv2.putText(
                    frame,
                    info_text,
                    (xmin, y_offset),
                    cv2.FONT_ITALIC,
                    0.5,  # You may want to use a smaller font size than the label_with_score
                    (
                        255,
                        0,
                        0,
                    ),
                    2,
                )
                # Move to the next line
                y_offset -= 20  # Adjust spacing based,font size and desired spacing


# Load the TFLite model and allocate tensors
# interpretation is a built in function in the library
interpreter = tflite.Interpreter(model_path="tflite-models/ssd_mobilenet.tflite")
interpreter.allocate_tensors()

# Get input and output details to display on camera window
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Check the input shape to resize images accordingly
input_shape = input_details[0]["shape"]

# Initialize camera
cap = cv2.VideoCapture(1)

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
    draw_boxes(frame, boxes, labels, classes, employment, scores)

    # Display the result
    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
