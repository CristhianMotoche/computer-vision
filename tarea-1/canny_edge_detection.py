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

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}")
        return

    threshold1 = 100
    threshold2 = 200

    def nothing(x):
        pass

    cv2.namedWindow('Canny Video')
    cv2.createTrackbar('Umbral 1 (min)', 'Canny Video', threshold1, 255, nothing)
    cv2.createTrackbar('Umbral 2 (max)', 'Canny Video', threshold2, 255, nothing)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        t1 = cv2.getTrackbarPos('Umbral 1 (min)', 'Canny Video')
        t2 = cv2.getTrackbarPos('Umbral 2 (max)', 'Canny Video')
        # Evitar que threshold1 sea mayor o igual a threshold2
        if t1 >= t2:
            edges = np.zeros_like(gray)
            contour_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.imshow('Canny Video', contour_img)
            cv2.displayOverlay('Canny Video', '¡Umbral 1 debe ser menor que Umbral 2!', 500)
        else:
            edges = cv2.Canny(gray, t1, t2)
            # Encontrar contornos en la imagen de bordes
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Dibujar los contornos sobre una imagen RGB para visualización
            contour_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 2)
            cv2.imshow('Canny Video', contour_img)
        key = cv2.waitKey(60) & 0xFF  # Espera 60 ms para ir más lento
        if key == 27:  # ESC para salir
            break
    cap.release()
    cv2.destroyAllWindows()

def process_image(url):
    try:
        resp = urllib.request.urlopen(url)
        image_data = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error: No se pudo cargar la imagen desde la URL {url}\n{e}")
        return
    if image is None:
        print(f"Error: No se pudo decodificar la imagen desde la URL {url}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold1_init = 100
    threshold2_init = 200

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.25)

    axs[0].set_title('1. Original')
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].axis('off')

    axs[1].set_title('2. Escala de grises')
    axs[1].imshow(gray, cmap='gray')
    axs[1].axis('off')

    edges = cv2.Canny(gray, threshold1_init, threshold2_init)
    # Encontrar contornos en la imagen de bordes
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Dibujar los contornos sobre una imagen RGB para visualización
    contour_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 2)
    im_edges = axs[2].imshow(contour_img)
    axs[2].set_title(f'3. Bordes y contornos\n(umbrales: {threshold1_init}, {threshold2_init})')
    axs[2].axis('off')

    axcolor = 'lightgoldenrodyellow'
    axthresh1 = plt.axes([0.25, 0.1, 0.5, 0.03], facecolor=axcolor)
    axthresh2 = plt.axes([0.25, 0.05, 0.5, 0.03], facecolor=axcolor)
    slider_thresh1 = Slider(axthresh1, 'Umbral 1 (min)', 0, 255, valinit=threshold1_init, valstep=1)
    slider_thresh2 = Slider(axthresh2, 'Umbral 2 (max)', 0, 255, valinit=threshold2_init, valstep=1)

    slider_thresh1.on_changed(lambda val: update(val, slider_thresh1, slider_thresh2, gray, axs, im_edges, fig))
    slider_thresh2.on_changed(lambda val: update(val, slider_thresh1, slider_thresh2, gray, axs, im_edges, fig))

    plt.show()


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
        # Encontrar contornos en la imagen de bordes
        contours, _ = cv2.findContours(edges_new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Dibujar los contornos sobre una imagen RGB para visualización
        contour_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 2)
        im_edges.set_data(contour_img)
        axs[2].set_title(f'3. Bordes y contornos\n(umbrales: {t1}, {t2})')
    fig.canvas.draw_idle()

def main():
    parser = argparse.ArgumentParser(description="Detección de Bordes Canny con OpenCV")
    parser.add_argument("--url", type=str, help="URL de la imagen de entrada")
    parser.add_argument("--video", type=str, help="Ruta al archivo de video local")
    args = parser.parse_args()

    if args.video:
        process_video(args.video)
    elif args.url:
        process_image(args.url)
    else:
        print("Debe proporcionar --url para imagen o --video para video.")

if __name__ == "__main__":
    main()
