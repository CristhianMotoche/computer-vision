SISTEMA DE RECONOCIMIENTO FACIAL
LBPH (Visión Clásica) + FaceNet (Deep Learning)

Este proyecto implementa un sistema de reconocimiento facial utilizando
dos enfoques diferentes de Visión por Computadora:

- LBPH (Local Binary Patterns Histograms), basado en visión clásica.
- FaceNet, basado en Deep Learning y embeddings faciales.

Ambos métodos utilizan el mismo conjunto de imágenes faciales, pero se
entrenan y ejecutan de manera independiente con el objetivo de permitir
una comparación justa de desempeño, precisión y costo computacional.


=====================================================================
REQUISITOS
=====================================================================

Este proyecto utiliza UV para la gestión de dependencias. Antes de ejecutar
el sistema, asegúrese de tener UV instalado:

    https://docs.astral.sh/uv/getting-started/installation/

Instalación de dependencias:

    uv sync

Para activar el entorno virtual:

    source .venv/bin/activate

O ejecutar comandos directamente con uv:

    uv run python script.py

Dependencias principales:
- opencv-python
- opencv-contrib-python  
- numpy
- mtcnn
- tensorflow
- keras-facenet


=====================================================================
EJECUCIÓN DEL SISTEMA
=====================================================================

Para que el sistema funcione correctamente, los scripts deben ejecutarse
ESTRICTAMENTE en el siguiente orden:

1) Captura inicial de imágenes faciales
--------------------------------------------------
    uv run python capture_faces.py

Este script abre la cámara web y solicita al usuario que gire el rostro
en distintas posiciones:
- frontal
- izquierda
- derecha
- arriba
- abajo

Las imágenes se guardan automáticamente y constituyen el dataset base
para el entrenamiento de los modelos biométricos.


2) Entrenamiento del modelo LBPH (Visión Clásica)
--------------------------------------------------
    uv run python extract_embeddings.py

En este paso, el sistema carga todas las imágenes capturadas y entrena
el modelo de reconocimiento facial LBPH (Local Binary Patterns Histograms).

Como resultado, se genera el archivo del modelo entrenado:
    *_lbph_model.yml

El modelo se guarda en la carpeta "models/".


3) Extracción de embeddings con FaceNet (Deep Learning)
--------------------------------------------------
    uv run python extract_facenet_embeddings.py

Este script utiliza FaceNet para extraer embeddings faciales a partir del
mismo dataset de imágenes capturadas previamente.

El resultado es un archivo con los embeddings faciales:
    usuario_facenet.npy

El archivo se guarda en la carpeta "models/".


4) Reconocimiento facial en tiempo real con LBPH
--------------------------------------------------
    uv run python realtime_match_lbph.py

Durante la ejecución:
- Se detecta el rostro en tiempo real.
- Se muestra el porcentaje de similitud (match %).
- Se indica si el acceso es PERMITIDO o DENEGADO.
- Cuando el acceso es permitido, se guarda automáticamente una imagen
  como evidencia.


5) Reconocimiento facial en tiempo real con FaceNet
--------------------------------------------------
    uv run python realtime_match_facenet.py

Durante la ejecución:
- Se detecta el rostro en tiempo real.
- Se muestra el porcentaje de similitud (match %).
- Se indica si el acceso es PERMITIDO o DENEGADO.
- Cuando el acceso es permitido, se guarda automáticamente una imagen
  como evidencia.


=====================================================================
NOTA FINAL
=====================================================================

Los modelos LBPH y FaceNet se ejecutan de manera independiente para
garantizar estabilidad del sistema y una evaluación experimental correcta.

Esto permite comparar ambos enfoques en términos de:
- Precisión
- Robustez ante cambios de iluminación y pose
- Velocidad de ejecución (FPS)
- Costo computacional

=====================================================================

