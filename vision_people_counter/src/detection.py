import cv2
import os
import numpy as np

# Chargement des fichiers du modèle
MODEL_PATH = os.path.join("models", "yolov3.weights")
CFG_PATH = os.path.join("models", "yolov3.cfg")
LABELS_PATH = os.path.join("models", "yolov3.txt")

with open(LABELS_PATH, "r") as f:
    CLASSES = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def start_detection():
    net = cv2.dnn.readNetFromDarknet(CFG_PATH, MODEL_PATH)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Impossible d'accéder à la webcam.")
        return

    print("✅ Webcam activée. Appuie sur 'q' pour quitter.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]

        # Préparation de l'image
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        ln = net.getUnconnectedOutLayersNames()
        layer_outputs = net.forward(ln)

        boxes, confidences, class_ids = [], [], []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5 and CLASSES[class_id] == "person":
                    box = detection[0:4] * np.array([width, height, width, height])
                    (centerX, centerY, w, h) = box.astype("int")

                    x = int(centerX - w / 2)
                    y = int(centerY - h / 2)

                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if len(indexes) > 0:
            for i in indexes.flatten():
                (x, y, w, h) = boxes[i]
                color = COLORS[class_ids[i]]
                label = f"{CLASSES[class_ids[i]]}: {confidences[i]:.2f}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Détection de personnes (OpenCV)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
