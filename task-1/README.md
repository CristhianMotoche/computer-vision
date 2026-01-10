# Canny Edge Detection

Este proyecto permite aplicar la detección de bordes de Canny tanto a imágenes (desde una URL) como a videos locales, con ajuste interactivo de umbrales.

## Requisitos

- Python 3.11+
- opencv-python
- matplotlib
- numpy
- uv (recomendado)

### Instalación de uv

Puedes instalar `uv` siguiendo las [instrucciones oficiales](https://docs.astral.sh/uv/getting-started/installation/):

```fish
curl -Ls https://astral.sh/uv/install.sh | sh
```

Luego, instala las dependencias del proyecto con:

```fish
uv pip install opencv-python matplotlib numpy
```

## Uso

### Procesar una imagen desde URL

```fish
uv run canny_edge_detection.py --url "https://purina.com.ec/sites/default/files/2025-09/Conoce-las-razas-de-gatos.jpg"
```

Aparecerá una ventana con la imagen original, en escala de grises y los bordes Canny. Puedes ajustar los umbrales con los sliders.

### Procesar un video local

```fish
uv run canny_edge_detection.py --video "/ruta/al/video.mp4"
```

Se abrirá una ventana mostrando los bordes Canny en tiempo real. Ajusta los umbrales con los sliders. Pulsa ESC para salir.

## Notas

- Solo uno de los argumentos (`--url` o `--video`) debe usarse por ejecución.
- Para videos, la reproducción es más lenta para facilitar el ajuste de umbrales.
- Se recomienda ejecutar los scripts usando [`uv`](https://github.com/astral-sh/uv) para un entorno más rápido y reproducible.

## Script: canny_edge_realtime.py

Este script permite aplicar la detección de bordes de Canny en tiempo real usando la cámara web.

### Uso

```fish
uv run canny_edge_realtime.py
```

Se abrirá una ventana mostrando la imagen original y los bordes detectados en tiempo real.

**Controles:**

- Presiona `q` para salir.
- Usa las teclas `w`/`s` para aumentar/disminuir el umbral 1 (mínimo).
- Usa las teclas `e`/`d` para aumentar/disminuir el umbral 2 (máximo).

El script previene automáticamente que el umbral 1 sea mayor o igual al umbral 2.

## Autor

- Basado en código de Canny y OpenCV, adaptado para uso interactivo.
