import cv2
from tflite_support.task import vision
from tflite_support.task import processor

model_path = "tflite-models/people-detection.tflite"

base_options = vision.Baseoptions(
    model_path=model_path,
    num_threads=2,
)
detection_options = vision.DetectionOptions(
    max_results=5,
    score_threshold=0.4,
)

detector = vision.ObjectDetectorOptions(base_options, detection_options)

model = vision.ObjectDetector.create_from_options(detector)


def bounding_boxes(frame):
    """Draw bounding boxes on the frame.
    Args:
        frame: the frame to draw on.
    """
    detections = model.process(frame)
    for detection in detections:
        frame = vision.draw_bounding_box_on_image(
            frame,
            detection.bbox,
            detection.class_name,
            detection.score,
            color=(0, 255, 0),
            thickness=2,
            display_str_list=[detection.class_name],
        )
    return frame


def getCaps():
    print("Getting camera")
    return cv2.VideoCapture(0)
