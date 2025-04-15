import cv2
from ultralytics import YOLO
import pyttsx3
import threading

# Initialize YOLOv11 (pretrained on COCO)
model = YOLO("yolo11n.pt")

# Initialize TTS engine
engine = pyttsx3.init( )
engine.setProperty('rate', 150)  # Speech speed (words per minute)
engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)

# Store detected objects to avoid duplicate announcements
detected_objects = set( )


def announce_objects():

    global detected_objects
    if detected_objects:
        announcement = "Detected: " + ", ".join(detected_objects)
        print(announcement)
        engine.say(announcement)
        engine.runAndWait( )
        detected_objects.clear( )


# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read( )
    if not ret:
        break

    # Run inference
    results = model(frame, verbose=False)  # Disable logs

    # Reset detected objects for this frame
    detected_objects = set( )

    # Process detections
    for result in results:
        for box in result.boxes:
            if box.conf > 0.8:  # Confidence threshold
                class_name = model.names[int(box.cls)]
                detected_objects.add(class_name)

    # Announce in a separate thread (non-blocking)
    threading.Thread(target=announce_objects).start( )

    # Display live detection
    cv2.imshow("YOLOv11 Detection", results[0].plot( ))

    if cv2.waitKey(1) == ord('q'):
        break

cap.release( )
cv2.destroyAllWindows( )