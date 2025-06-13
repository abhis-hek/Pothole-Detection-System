from ultralytics import YOLO
import cv2
import math 
import requests

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("best.pt")

# object classes
classNames = ["pothole"]

# Flag to keep track of pothole detection
pothole_detected = False

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # Reset the flag before processing each frame
    pothole_detected = False

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->", confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            # display confidence percentage
            cv2.putText(img, f"{classNames[cls]} {confidence * 100}%", org, font, fontScale, color, thickness)

            # Update the flag if a pothole is detected
            if classNames[cls] == "pothole":
                pothole_detected = True

    # Control the LED based on pothole detection
    esp32_ip = "192.168.209.41"  # replace with the actual IP address of your ESP32

    if pothole_detected:
        response = requests.get(f"http://{esp32_ip}/ledon")
    else:
        response = requests.get(f"http://{esp32_ip}/ledoff")

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
