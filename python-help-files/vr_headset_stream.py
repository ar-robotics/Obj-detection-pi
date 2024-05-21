from flask import Flask, Response
import cv2

app = Flask(__name__)

# Initialize video capture with the first webcam device
cap = cv2.VideoCapture("/dev/video0")


def generateframe():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        # If frame is not captured successfully, return an empty frame
        return b""

    # Encode frame as JPEG
    ret, buffer = cv2.imencode(".jpg", frame)
    if not ret:
        # If frame is not encoded successfully, return an empty frame
        return b""

    # Convert to bytes and return
    return buffer.tobytes()


@app.route("/snapshot")
def snapshot():
    frame = generateframe()
    return Response(frame, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
