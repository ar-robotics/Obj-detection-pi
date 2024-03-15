import cv2

cap = cv2.VideoCapture(1)
# 0 is  the default camera

while True:
    capture, frame = (
        cap.read()
    )  # Read the frame, capture is a boolean if fram was captured or not
    if capture:
        cv2.imshow("Camera Feed", frame)  # displays frame on a window
        if cv2.waitKey(1) & 0xFF == ord("q"):  # Exit on pressing 'q'
            break
    else:
        break

cap.release()  # releases the camera
cv2.destroyAllWindows()
