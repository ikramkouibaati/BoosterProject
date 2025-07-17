import cv2
from ultralytics import YOLO

# Charger le modèle YOLOv5s (personnes uniquement)
model = YOLO("yolov5s.pt")

# Zone d'intérêt (ROI) fixe — à adapter selon ta vidéo
roi_x, roi_y, roi_w, roi_h = 200, 100, 300, 300

def is_in_roi(cx, cy):
    return roi_x <= cx <= roi_x + roi_w and roi_y <= cy <= roi_y + roi_h

# Webcam ou fichier vidéo
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Appliquer la détection
    results = model(frame, verbose=False)[0]

    count_in_roi = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        # Vérifie que la classe détectée est "person"
        if model.names[cls_id] == 'person' and conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if is_in_roi(cx, cy):
                count_in_roi += 1
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 5, color, -1)

    # Rectangle ROI
    roi_x, roi_y, roi_w, roi_h = 100, 50, 600, 500
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 255, 0), 2)
    cv2.putText(frame, f"In ROI: {count_in_roi}", (roi_x, roi_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.namedWindow("YOLOv5 + ROI", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("YOLOv5 + ROI", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow("YOLOv5 + ROI", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
