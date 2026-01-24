import os
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet

# ================= CONFIG =================
PERSON_NAME = "usuario"
DATASET_DIR = f"dataset/{PERSON_NAME}"
OUTPUT_PATH = f"models/{PERSON_NAME}_facenet.npy"
os.makedirs("models", exist_ok=True)

# ================= MODELOS =================
detector = MTCNN()
embedder = FaceNet()

embeddings = []

print("Cargando imágenes del dataset (FaceNet)...")

for img_name in os.listdir(DATASET_DIR):
    img_path = os.path.join(DATASET_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    if len(faces) == 0:
        continue

    x, y, w, h = faces[0]["box"]
    face = rgb[y:y+h, x:x+w]

    face = cv2.resize(face, (160, 160))
    face = np.expand_dims(face, axis=0)

    embedding = embedder.embeddings(face)[0]
    embeddings.append(embedding)

print(f"Total de embeddings generados: {len(embeddings)}")

embeddings = np.array(embeddings)
np.save(OUTPUT_PATH, embeddings)

print("Embeddings FaceNet guardados correctamente")
print(f"Archivo generado: {OUTPUT_PATH}")
