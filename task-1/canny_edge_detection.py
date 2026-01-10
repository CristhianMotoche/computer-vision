# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "opencv-python",
# ]
# ///
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import urllib.request
from matplotlib.widgets import Slider


# Función para actualizar la imagen de bordes Canny
def update(val, slider_thresh1, slider_thresh2, gray, axs, im_edges, fig):
    t1 = int(slider_thresh1.val)
    t2 = int(slider_thresh2.val)
    # Evitar que threshold1 sea mayor o igual a threshold2
    if t1 >= t2:
        axs[2].set_title('¡Umbral 1 debe ser menor que Umbral 2!')
        im_edges.set_data(np.zeros_like(gray))
    else:
        edges_new = cv2.Canny(gray, t1, t2)
        im_edges.set_data(edges_new)
        axs[2].set_title(f'3. Bordes Canny\n(umbrales: {t1}, {t2})')
    fig.canvas.draw_idle()

def main():
    parser = argparse.ArgumentParser(description="Detección de Bordes Canny con OpenCV")
    parser.add_argument("--url", type=str, required=True, help="URL de la imagen de entrada")
    args = parser.parse_args()

    # Cargar imagen desde la URL
    try:
        resp = urllib.request.urlopen(args.url)
        image_data = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error: No se pudo cargar la imagen desde la URL {args.url}\n{e}")
        return
    if image is None:
        print(f"Error: No se pudo decodificar la imagen desde la URL {args.url}")
        return

    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Explicación de los thresholds:
    # threshold1: Valor mínimo. Los gradientes de píxeles por debajo de este valor se descartan como no bordes.
    # threshold2: Valor máximo. Los gradientes de píxeles por encima de este valor se consideran bordes definitivos.
    # Los valores entre threshold1 y threshold2 se consideran bordes solo si están conectados a un borde fuerte.

    # Valores iniciales de los umbrales
    threshold1_init = 100
    threshold2_init = 200

    # Crear la figura y los ejes
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.25)

    # Mostrar imagen original
    axs[0].set_title('1. Original')
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].axis('off')

    # Mostrar imagen en escala de grises
    axs[1].set_title('2. Escala de grises')
    axs[1].imshow(gray, cmap='gray')
    axs[1].axis('off')

    # Mostrar bordes Canny iniciales
    edges = cv2.Canny(gray, threshold1_init, threshold2_init)
    im_edges = axs[2].imshow(edges, cmap='gray')
    axs[2].set_title(f'3. Bordes Canny\n(umbrales: {threshold1_init}, {threshold2_init})')
    axs[2].axis('off')

    # Crear sliders para los thresholds
    axcolor = 'lightgoldenrodyellow'
    axthresh1 = plt.axes([0.25, 0.1, 0.5, 0.03], facecolor=axcolor)
    axthresh2 = plt.axes([0.25, 0.05, 0.5, 0.03], facecolor=axcolor)
    slider_thresh1 = Slider(axthresh1, 'Umbral 1 (min)', 0, 255, valinit=threshold1_init, valstep=1)
    slider_thresh2 = Slider(axthresh2, 'Umbral 2 (max)', 0, 255, valinit=threshold2_init, valstep=1)

    # Conectar los sliders con la función update definida fuera de main
    slider_thresh1.on_changed(lambda val: update(val, slider_thresh1, slider_thresh2, gray, axs, im_edges, fig))
    slider_thresh2.on_changed(lambda val: update(val, slider_thresh1, slider_thresh2, gray, axs, im_edges, fig))

    plt.show()

if __name__ == "__main__":
    main()
