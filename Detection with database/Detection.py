import cv2
import numpy as np
import importlib.util

pkg = importlib.util.find_spec("tflite_runtime")
if pkg in locals():
    import tflite_runtime.interpreter as tflite
else:
    import tensorflow.lite.Interpreter as tflite


class ExtractModel:
    """Class for extracting the TFLite model and labels from the file.

    Attributes:
        model_path: Path to the TFLite model
        label_path: Path to the label file
        interpreter: TFLite interpreter
    """

    def __init__(self, model_path, label_path):
        """Initialize the TFLite interpreter

        Args:
            model_path: Path to the TFLite model
            label_path: Path to the label file
        """
        self.model_path = model_path
        self.__interpreter = tflite(model_path=self.model_path)
        self.label_path = label_path
        self.__interpreter.allocate_tensors()

    def load_labels(self):
        """Load the labels from the label file

        Returns:
            labels: List of labels
        """
        with open(self.label_path, "r") as f:
            return [line.strip() for line in f.readlines()]

    def get_interpreter(self):
        """Get the TFLite interpreter

        Returns:
            interpreter: TFLite interpreter
        """
        return self.__interpreter

    def get_details(self):
        """Get the input and output details of the TFLite model

        Returns:
            input_details: Input details
            output_details: Output details
            height: Height of the input
            width: Width of the input
        """
        input_details = self.__interpreter.get_input_details()
        output_details = self.__interpreter.get_output_details()
        height = int(input_details[0]["shape"][1])
        width = int(input_details[0]["shape"][2])
        return input_details, output_details, height, width


class Detection:
    """Class for object detection using TFLite model

    Attributes:
        labels: List of labels
        height: Height of the input
        width: Width of the input
        interpreter: TFLite interpreter
        input_detail: Input details
        output_detail: Output details
        collection: Collection object
    """

    def __init__(
        self,
        labels,
        height,
        width,
        interpreter,  # noqa
        input_detail,
        output_detail,
        collection,
        input_data=None,
        imW=1280,
        imH=720,
    ):
        self.frame = None
        self.imW = imW
        self.imH = imH
        self.height = height
        self.width = width
        self.labels = labels
        self.interpreter = interpreter
        self.input_detail = input_detail
        self.output_detail = output_detail
        self.scores = None
        self.boxes = {}
        self.classes = None
        self.frame_rate_calc = None
        self.freq = None
        self.num_detections = None
        self.collection = collection
        self.boxes_idx, self.classes_idx, self.scores_idx = self.set_index()

    def set_index(self):
        """Set the index of the boxes, classes, and scores from output details.


        Returns:
            boxes_idx: Index of the boxes in the output details
            classes_idx: Index of the classes in the output details
            scores_idx: Index of the scores in the output details
        """
        # determine if the model is tf1 or tf2
        # because outputs are ordered differently for TF2 and TF1 models
        outname = self.output_detail[0]["name"]
        if "StatefulPartitionedCall" in outname:  # This is a TF2 model
            boxes_idx, classes_idx, scores_idx = 1, 3, 0
        else:  # This is a TF1 model
            boxes_idx, classes_idx, scores_idx = 0, 1, 2
        return boxes_idx, classes_idx, scores_idx

    def interpret(self) -> None:
        """Interpret the frame and make predictions

        Args:
            frame: Input frame
            height: Frame height
            width: Frame width
            interpreter: TFLite interpreter
            input_details: Input details
            output_details: Output details

        Returns:
            num_detections: Number of detections
            scores: Confidence scores
            boxes: Bounding box coordinates
            classes: Class indices
        """
        frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.width, self.height))
        input_data = np.expand_dims(frame_resized, axis=0)
        input_data = self.__floating_model(input_data)
        self.interpreter.set_tensor(self.input_detail[0]["index"], input_data)
        self.interpreter.invoke()
        # self.scores = self.interpreter.get_tensor(  # noqa
        #     self.output_detail[self.scores_idx]["index"]
        # )[0]
        # self.boxes = self.interpreter.get_tensor(  # noqa
        #     self.output_detail[self.boxes_idx]["index"]
        # )[0]
        # self.classes = self.interpreter.get_tensor(  # noqa
        #     self.output_detail[self.classes_idx]["index"]
        # )[0]
        output_data = self.interpreter.get_tensor(
            self.output_detail[0]["index"]
        )  # noqa
        print(output_data)

    def __floating_model(self, input_data):
        """Normalize input data to float type

        Args:
            input_data: Input data
            input_details: Input details

        Returns:
            input_data: Normalized input data
        """
        floating_model = self.input_detail[0]["dtype"] == np.float32

        input_mean = 127.5
        input_std = 127.5
        # floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std
            return input_data
        else:
            return input_data

    def __retreive_info(self, object_name) -> dict:
        """
        Retrieve all information for the object from the database

        Args:
            database: Database object
            object_name: Name of the object

        Returns:
            all_info_dict: Dictionary containing all information for the object
        """
        query = {"type": object_name}
        items_info = self.collection.find_one(query)
        all_info_dict = {}
        if items_info:
            for key, value in items_info.items():
                # Save each key-value pair into the dictionary
                if key != "_id" and key != "type":
                    all_info_dict[key] = value
        return all_info_dict

    def analyze(self, frame1):
        self.frame = frame1.copy()
        self.interpret()

        processed_frame = self.make_boxes()
        return processed_frame

    def draw_framerate(self, frame_rate_calc):
        # Draw framerate in corner of frame
        cv2.putText(
            self.frame,
            "FPS: {0:.2f}".format(frame_rate_calc),
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    def make_boxes(self):
        """Draw bounding boxes on the frame and display the information

        Args:
            ImH: Height of the frame
            ImW: Width of the frame

        Returns:
            frame: Frame with bounding boxes
        """
        for i in range(len(self.scores)):
            if (self.scores[i] > 0.6) and (
                self.scores[i] <= 1.0
            ):  # Confidence threshold

                ymin = int(max(1, (self.boxes[i][0] * self.imH)))
                xmin = int(max(1, (self.boxes[i][1] * self.imW)))
                ymax = int(min(self.imH, (self.boxes[i][2] * self.imH)))
                xmax = int(min(self.imW, (self.boxes[i][3] * self.imW)))
                cv2.rectangle(
                    self.frame,
                    (xmin, ymin),
                    (xmax, ymax),
                    (10, 255, 0),
                    2,  # noqa
                )
                object_name = self.labels[int(self.classes[i])]
                dict_info = self.__retreive_info(object_name)
                # Loop through all key-value pairs in the items_info document
                label_with_score = "{}: {:.2f}".format(
                    object_name, self.scores[i]  # noqa
                )
                labelSize, baseLine = cv2.getTextSize(
                    label_with_score, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )  # Get font size
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(
                    self.frame,
                    (xmin, label_ymin - labelSize[1] - 10),
                    (xmin + labelSize[0], label_ymin + baseLine - 10),
                    (255, 255, 255),
                    cv2.FILLED,
                )  # Draw white box to put label text in
                y_offset = ymin - 25
                cv2.putText(
                    self.frame,
                    label_with_score,
                    (xmin, label_ymin - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                )  # Draw label text
                if dict_info:
                    for key, value in dict_info.items():
                        # Generate the text for this key-value pair
                        info_text = "{}: {}".format(key, value)
                        # Display this key-value pair on the frame
                        cv2.putText(
                            self.frame,
                            info_text,
                            (int(xmin), int(y_offset)),
                            cv2.FONT_ITALIC,
                            0.5,  # Yo
                            (
                                255,
                                0,
                                0,
                            ),
                            2,
                        )
                        y_offset -= 20
        return self.frame

    def calculate_framerate(self):
        """Calculate and return the frame rate.

        This function calculates the frame rate by
        dividing the number of frames processed
        by the time taken to process those frames.
        The frame rate is returned along with
        the frequency of the system clock.

        Returns:
            tuple: A tuple containing the frame rate and the clock frequency.

        """
        self.frame_rate_calc = 1
        self.freq = cv2.getTickFrequency()
        return self.frame_rate_calc, self.freq
