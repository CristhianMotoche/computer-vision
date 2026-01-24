import cv2
import time
import os

# ===============================
# CONFIGURACIÓN
# ===============================
WINDOW_NAME = "Biometrico Alta Precision - Grupo 8"
MODEL_PATH = "models/usuario_lbph_model.yml"

THRESHOLD = 400        # umbral LBPH
ACCESS_MATCH = 50      # % mínimo para permitir acceso

CAPTURE_DIR = "captures"
os.makedirs(CAPTURE_DIR, exist_ok=True)

CAPTURE_COOLDOWN = 3   # segundos entre capturas
last_capture_time = 0

# ===============================
# CARGAR MODELO
# ===============================
model = cv2.face.LBPHFaceRecognizer_create()
model.read(MODEL_PATH)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
prev_time = time.time()

# ===============================
# LOOP PRINCIPAL
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (224, 224))
        face_img = cv2.equalizeHist(face_img)

        # ===============================
        # PREDICCIÓN
        # ===============================
        label, confidence = model.predict(face_img)

        # ===============================
        # CÁLCULO DE MATCH
        # ===============================
        if confidence < THRESHOLD:
            match = int(100 * (1 - confidence / THRESHOLD))
        else:
            match = 0

        # ===============================
        # CONTROL DE ACCESO
        # ===============================
        if match >= ACCESS_MATCH:
            status = "ACCESO PERMITIDO"
            color = (0, 255, 0)

            # ===============================
            # CAPTURA AUTOMÁTICA
            # ===============================
            current_time = time.time()
            if current_time - last_capture_time > CAPTURE_COOLDOWN:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"{CAPTURE_DIR}/acceso_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                last_capture_time = current_time

        else:
            status = "ACCESO DENEGADO"
            color = (0, 0, 255)

        # ===============================
        # DIBUJO EN PANTALLA
        # ===============================
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        cv2.putText(
            frame,
            f"Match: {match}%",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )

        cv2.putText(
            frame,
            status,
            (x, y + h + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            3
        )

    # ===============================
    # FPS
    # ===============================
    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    cv2.imshow(WINDOW_NAME, frame)

    # ESC o Q para salir
    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

# ===============================
# CIERRE
# ===============================
cap.release()
cv2.destroyAllWindows()
