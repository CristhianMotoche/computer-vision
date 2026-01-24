import cv2
import os
import time

# Configuración
PERSON_NAME = "usuario"
OUTPUT_DIR = f"dataset/{PERSON_NAME}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

poses = ["frontal", "left", "right", "up", "down"]
pose_index = 0
img_count = 0

# Cargar clasificador Haar
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
last_capture = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    if len(faces) > 0:
        (x, y, w, h) = faces[0]

        # Dibujar bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Texto de instrucción
        cv2.putText(
            frame,
            f"Mira: {poses[pose_index]}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        # Guardar imagen cada 2 segundos
        if time.time() - last_capture > 2:
            img_name = f"{poses[pose_index]}_{img_count}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), frame)
            print("Imagen guardada:", img_name)
            img_count += 1
            last_capture = time.time()

            if img_count % 5 == 0:
                pose_index += 1
                if pose_index >= len(poses):
                    break

    cv2.imshow("Captura Facial", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
