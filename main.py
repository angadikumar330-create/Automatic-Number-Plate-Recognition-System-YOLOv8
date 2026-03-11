import cv2
from tracker import Tracker

# Open video
cap = cv2.VideoCapture("highway.mp4")

# Object detector
object_detector = cv2.createBackgroundSubtractorMOG2()

# Tracker
tracker = Tracker()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    height, width, _ = frame.shape

    # Define Region of Interest
    roi = frame[300:720, 400:900]

    # Detect moving objects
    mask = object_detector.apply(roi)

    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detections = []

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > 100:

            x, y, w, h = cv2.boundingRect(cnt)

            detections.append([x, y, w, h])

    # Tracking
    boxes_ids = tracker.update(detections)

    for box_id in boxes_ids:

        x, y, w, h, id = box_id

        # Add ROI offset so boxes align correctly
        x = x + 400
        y = y + 300

        cv2.putText(frame, str(id), (x, y - 15),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Draw ROI area
    cv2.rectangle(frame, (400,300), (900,720), (255,0,0), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
