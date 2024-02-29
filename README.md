# Object_Detection on Raspberry pi 5
To deploy this on the raspberry pi:
1.Create a virtual environment:
```python
    python -m venv myenv
    source myenv/bin/activate
```
2. download requirements:
```python,
    pip install -r requirements.txt
```
3. go to either pre-trained model or custom-transfer learning directory using cd
4. Run test.py


## Pre-trained model directory

The code uses the COCO dataset and a pre-trained model called mobilenet to detect objects from the dataset.
Link for the coco dataset:  https://cocodataset.org/#home 
Link for the Mobilenet tflite model : https://www.kaggle.com/models/iree/ssd-mobilenet-v2
you can also stream the object detection, which the VR headset displays.

## Custom-transfer learning

Here, I followed the tutorial for transfer learning on a custom dataset given my Tensorflow :
https://www.tensorflow.org/lite/models/modify/model_maker/object_detection 
In the directory tflite_models there are several Tensorflow lite models to choose from.
The best one for now is the people_Detection_2 one.
The datasets I trained on : https://www.kaggle.com/datasets/sbaghbidi/human-faces-object-detection?rvi=1

‘Open Images Dataset V7’. Accessed: Feb. 23, 2024. [Online]. Available: https://storage.googleapis.com/openimages/web/visualizer/index.html?type=detection&set=train&c=%2Fm%2F02rdsp
