import cv2
import os
import numpy as np

DATASET_DIR = "dataset/usuario"
MODEL_PATH = "models/usuario_lbph_model.yml"

faces = []
labels = []
label = 0

print("Cargando imágenes del dataset (todas las poses)...")

for file in os.listdir(DATASET_DIR):
    if not file.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(DATASET_DIR, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        continue

    img = cv2.resize(img, (224, 224))
    img = cv2.equalizeHist(img)

    faces.append(img)
    labels.append(label)

faces = np.array(faces)
labels = np.array(labels)

print(f"Total de imágenes usadas: {len(faces)}")

if len(faces) == 0:
    raise ValueError("No se cargaron imágenes. Revisa el dataset.")

model = cv2.face.LBPHFaceRecognizer_create(
    radius=2,
    neighbors=16,
    grid_x=8,
    grid_y=8
)

model.train(faces, labels)

# 🔴 GUARDADO EXPLÍCITO
model.save(MODEL_PATH)

# 🔴 CONFIRMACIÓN
if os.path.exists(MODEL_PATH):
    print("===================================")
    print("Modelo LBPH entrenado y guardado correctamente")
    print(f"Archivo generado: {MODEL_PATH}")
else:
    print("ERROR: el modelo NO se guardó")
