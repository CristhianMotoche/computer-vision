import cv2
import numpy as np
import os
import time
from mtcnn import MTCNN
from keras_facenet import FaceNet

# =========================
# CONFIGURACIÓN GENERAL
# =========================
LBPH_MODEL_PATH = "models/usuario_lbph_model.yml"
FACENET_EMB_PATH = "models/usuario_facenet.npy"
CAPTURES_DIR = "captures"
THRESHOLD_LBPH = 65       # % mínimo para aprobar
THRESHOLD_FACENET = 0.65  # distancia coseno máxima

os.makedirs(CAPTURES_DIR, exist_ok=True)

# =========================
# CARGA MODELOS
# =========================
print("Cargando modelos...")

# LBPH
lbph = cv2.face.LBPHFaceRecognizer_create()
lbph.read(LBPH_MODEL_PATH)

# FaceNet
embedder = FaceNet()
known_embeddings = np.load(FACENET_EMB_PATH)

# Detector facial
detector = MTCNN()

# =========================
# UTILIDADES
# =========================
def lbph_match(face_gray):
    label, confidence = lbph.predict(face_gray)
    match = max(0, min(100, int(100 - confidence)))
    return match, match >= THRESHOLD_LBPH


def facenet_match(face_rgb):
    face_rgb = cv2.resize(face_rgb, (160, 160))
    emb = embedder.embeddings(np.expand_dims(face_rgb, axis=0))[0]

    emb = emb / np.linalg.norm(emb)
    known = known_embeddings / np.linalg.norm(known_embeddings, axis=1, keepdims=True)

    similarities = np.dot(known, emb)
    best = np.max(similarities)

    return best * 100, best >= THRESHOLD_FACENET


# =========================
# CÁMARA
# =========================
cap = cv2.VideoCapture(0)
prev_time = time.time()
captured = False

print("Iniciando comparativo LBPH vs FaceNet...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    # FPS
    now = time.time()
    fps = int(1 / (now - prev_time))
    prev_time = now
    cv2.putText(frame, f"FPS: {fps}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    for face in faces:
        x, y, w, h = face["box"]
        x, y = abs(x), abs(y)

        face_rgb = rgb[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
        face_gray = cv2.resize(face_gray, (224, 224))
        face_gray = cv2.equalizeHist(face_gray)

        lbph_score, lbph_ok = lbph_match(face_gray)
        facenet_score, facenet_ok = facenet_match(face_rgb)

        # Texto
        lbph_text = f"LBPH: {lbph_score:.1f}% [{'OK' if lbph_ok else 'NO'}]"
        fn_text = f"FaceNet: {facenet_score:.1f}% [{'OK' if facenet_ok else 'NO'}]"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, lbph_text, (x, y-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, fn_text, (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Captura automática (solo una vez)
        if (lbph_ok or facenet_ok) and not captured:
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"{CAPTURES_DIR}/access_{ts}.jpg", frame)
            captured = True

    cv2.imshow("Biometrico Comparativo - LBPH vs FaceNet", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
