import cv2


cap = cv2.VideoCapture(0)

for i in range(137,155):
    
    
    ret, frame = cap.read()
    if ret: 
        cv2.imwrite(f"image_{i}.jpg", frame)
        cv2.waitKey(50)
        
        
cap.release()
cv2.destroyAllWindows()