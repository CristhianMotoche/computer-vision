import os
# Silenciar logs de TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import time
from mtcnn import MTCNN
from keras_facenet import FaceNet
from numpy.linalg import norm

# ================= CONFIGURACIÓN =================
PERSON_NAME = "usuario"   # debe coincidir con el archivo .npy
MODEL_PATH = f"models/{PERSON_NAME}_facenet.npy"
ACCESS_THRESHOLD = 0.6    # más alto = más estricto
SAVE_ON_ACCESS = True

# ================= CARGA DE MODELOS =================
known_embeddings = np.load(MODEL_PATH)

detector = MTCNN()
embedder = FaceNet()

cap = cv2.VideoCapture(0)

# Flag para evitar múltiples capturas
access_granted = False

# ================= FUNCIONES =================
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# ================= LOOP PRINCIPAL =================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    # Si no hay rostro, resetear acceso
    if len(faces) == 0:
        access_granted = False

    for face_data in faces:
        x, y, w, h = face_data["box"]
        face = rgb[y:y+h, x:x+w]

        if face.size == 0:
            continue

        face = cv2.resize(face, (160, 160))
        face = np.expand_dims(face, axis=0)

        # Obtener embedding FaceNet (sin verbose)
        emb = embedder.embeddings(face)[0]

        sims = [cosine_similarity(emb, e) for e in known_embeddings]
        best_sim = max(sims)

        match_percent = max(0, min(100, best_sim * 100))
        access = best_sim >= ACCESS_THRESHOLD

        color = (0, 255, 0) if access else (0, 0, 255)
        label = "ACCESO PERMITIDO" if access else "ACCESO DENEGADO"

        # Dibujar resultados
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"Match: {match_percent:.1f}%",
                    (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, label,
                    (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 📸 Captura SOLO una vez cuando se aprueba acceso
        if access and not access_granted and SAVE_ON_ACCESS:
            filename = f"access_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            access_granted = True

    cv2.imshow("Biometrico Alta Precision - FaceNet", frame)

    # Salir con ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
