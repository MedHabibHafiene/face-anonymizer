import cv2
import mediapipe as mp

webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
) as face_detection:
    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Failed to grab frame.")
            break

        height, width, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = face_detection.process(img_rgb)

        if out.detections is not None:
            for detection in out.detections:
                bbox = detection.location_data.relative_bounding_box
                x1, y1, w, h = (
                    int(bbox.xmin * width),
                    int(bbox.ymin * height),
                    int(bbox.width * width),
                    int(bbox.height * height),
                )

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x1 + w)
                y2 = min(height, y1 + h)

                frame[y1:y2, x1:x2] = cv2.blur(
                    frame[y1:y2, x1:x2],
                    (30, 30),
                )

        cv2.imshow("Face Anonymizer", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

webcam.release()
cv2.destroyAllWindows()
