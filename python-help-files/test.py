import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
from PIL import Image

def test_with_an_image(modelpath,imagepath,labelpath):
    
    # Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Read the image and preprocess it
    image = Image.open(imagepath)

    image = image.resize((input_details[0]['shape'][2], input_details[0]['shape'][1]))
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.uint8)

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], image)

    # Run the inference
    interpreter.invoke()

    # Extract the output data
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_class_index = np.argmax(output_data)
    predicted_class_score = np.max(output_data)
    print(f"Predicted class index: {predicted_class_index}, Score: {predicted_class_score}")

    with open(labelpath, 'r') as file:
        labels = [line.strip() for line in file.readlines()]
    predicted_class_name = labels[predicted_class_index]
    print(f"Predicted class name: {predicted_class_name}")


def live_detection(modelpath,labelpath):
    # Load TFLite model and allocate tensors
    interpreter = tflite.Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    #print(output_details)
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # Load labels
    with open(labelpath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Initialize video capture
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame to model input dimensions
        input_frame = cv2.resize(frame, (width, height))
        
        # Convert frame to float32 and normalize (if your model requires normalization)
        input_frame = np.expand_dims(input_frame, axis=0).astype(np.uint8)
        # input_frame /= 255.0  # Uncomment this if your model expects input values to be normalized
        
        # Set the model input and run inference
        interpreter.set_tensor(input_details[0]['index'], input_frame)
        interpreter.invoke()
        
        # Retrieve detection results
        num_detections = int(interpreter.get_tensor(output_details[2]['index'])[0])
        
        scores = interpreter.get_tensor(output_details[0]['index'])[0]
        #print(scores.shape)# Confidence scores
        boxes = interpreter.get_tensor(output_details[1]['index'])[0] 
        #print(boxes.shape)# Bounding box coordinates
        classes = interpreter.get_tensor(output_details[3]['index'])[0]  # Class inde
        #print(classes.shape)
        # Iterate over detections and draw bounding boxes on the original frame
        for i in range(num_detections):
            if scores[i] > 0.4:  # Confidence threshold
                ymin, xmin, ymax, xmax = boxes[i]
                (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                            ymin * frame.shape[0], ymax * frame.shape[0])
                cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
                
                # Draw label
                object_name = labels[int(classes[i])]  # Retrieve the class name
                label = '%s' % (object_name)  # Example: 'cat: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
                top = max(top, labelSize[1])
                cv2.rectangle(frame, (int(left), int(top - round(1.5*labelSize[1]))), (int(left + round(1.5*labelSize[0])), int(top + baseLine)), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Object Detection', frame)
        
        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release capture
    cap.release()
    cv2.destroyAllWindows()
    
    modelpath = 'tflite_models/people_detection_2.tflite'
    labelpath = 'labels.txt'
    live_detection(modelpath,labelpath)