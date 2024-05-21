from flask import Flask, Response
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils import visualize, Camera
from jsondatabase import JsonDatabase

app = Flask(__name__)
COUNTER, FPS = 0, 0
START_TIME = time.time()
# Initialize global variables for the object detection
model_path = "/home/pi/Obj-detection-pi/custom-transfer-learning/tflite_models/bolt-detection-mp.tflite"


class ObjectDetection:
    """ObjectDetection class for the object detection application"""

    def __init__(self, model_path):
        self.row_size = 50  # pixels
        self.left_margin = 24  # pixels
        self.text_color = (46, 26, 119)  #
        self.db_text_color = (57, 25, 215)  # green
        self.font_size = 1
        self.font_thickness = 1
        self.fps_avg_frame_count = 10
        self.model_path = model_path
        self.detection_frame = None
        self.detection_result_list = []
        self.current_camera = Camera.DRIVE
        self.cap = cv2.VideoCapture(self.current_camera.value)
        self.db = JsonDatabase("data.json")
        self.dict_info = None

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

        Returns:
            None

        """
        global FPS, COUNTER, START_TIME

        # Calculate the FPS
        if COUNTER % self.fps_avg_frame_count == 0:
            FPS = self.fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()

        self.detection_result_list.append(result)
        COUNTER += 1

    def inference(self, max_results=5, score_threshold=0.7):
        """Perform the object detection inference

        Args:
            max_results (int): The maximum number of results
            score_threshold (float): The score threshold

        Returns:
            detector (vision.ObjectDetector): The object detector
        """
        # Create the base options for detector
        base_options = python.BaseOptions(model_asset_path=model_path)

        # Create the object detector options
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            max_results=max_results,
            score_threshold=score_threshold,
            result_callback=self.save_result,
        )
        # Create the object detector
        detector = vision.ObjectDetector.create_from_options(options)
        return detector

    def _detect_objects(self, frame, detector):
        """Detect objects in the frame

        Args:
            frame (np.ndarray): The input frame
            detector (vision.ObjectDetector): The object detector

        Returns:
            np.ndarray: The frame with the detected objects
        """
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
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

            frame, category, text_location = visualize(
                frame, self.detection_result_list[0]
            )
            self.dict_info = self.__retreive_info(category)

            self.detection_result_list.clear()

        if self.dict_info:
            offset = text_location[1] - 35
            for key, value in self.dict_info.items():
                info_text = "{}: {}".format(key, value)
                cv2.putText(
                    frame,
                    info_text,
                    (text_location[0], offset),
                    cv2.FONT_HERSHEY_DUPLEX,
                    self.font_size,
                    self.db_text_color,
                    self.font_thickness,
                    cv2.LINE_AA,
                )
                offset -= int(self.font_size * 20)
        return frame

    def generate_frames(self, detector):
        """Generate the frames for the object detection's snapshot

        Args:
            detector (vision.ObjectDetector): The object detector

        Returns:
            bytes: The frame in bytes
        """
        ret, frame = self.cap.read()
        if not ret:
            return b""
        frame = self._detect_objects(frame, detector)
        detection_frame = frame
        ret, buffer = cv2.imencode(".jpg", detection_frame)
        if not ret:
            return b""
        detection_frame = buffer.tobytes()
        return detection_frame

    def __retreive_info(self, object_name) -> dict:
        """
        Retrieve all information for the object from the database

        Args:
            database: Database object
            object_name: Name of the object

        Returns:
            all_info_dict: Dictionary containing all information for the object
        """
        if object_name is not None:
            object_name = object_name.lower()
        query = {"class": object_name}
        items_info = self.db.find_one(query)
        all_info_dict = {}

        if items_info:
            for key, value in items_info.items():
                # Save each key-value pair into the dictionary
                if key != "_id" and key != "type":
                    all_info_dict[key] = value

        return all_info_dict

    def gen_videostream(self, detector):
        """Generate the video stream for the object detection

        Args:
            detector (vision.ObjectDetector): The object detector


        Yields:
            bytes: The frame in bytes
        """
        while (self.cap).isOpened():
            ret, frame = self.cap.read()
            if not ret:
                return b""
            else:
                frame = self._detect_objects(frame, detector)
                detection_frame = frame
                ret, buffer = cv2.imencode(".jpg", detection_frame)
                detection_frame = buffer.tobytes()
                if not ret:
                    # If frame is not encoded successfully, return an empty frame
                    return b""

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + detection_frame + b"\r\n"
                )


Det_object = ObjectDetection(model_path)
detector = Det_object.inference(3, 0.8)


@app.route("/video_feed")
def video_feed():
    return Response(
        Det_object.gen_videostream(detector),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# @app.route("/snapshot")
# def snapshot():
#     frame = Det_object.generate_frames(detector)
#     if frame is None
#         return "Failed to capture frame", 400
#     return Response(frame, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
