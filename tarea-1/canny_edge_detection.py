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

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(bottom=0.25)

    # 1. Imagen original
    axs[0, 0].set_title('1. Original')
    axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0, 0].axis('off')

    # 2. Imagen en escala de grises
    axs[0, 1].set_title('2. Escala de grises')
    axs[0, 1].imshow(gray, cmap='gray')
    axs[0, 1].axis('off')

    # 3. Bordes y contornos (sin ruido)
    edges = cv2.Canny(gray, threshold1_init, threshold2_init)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 2)
    im_edges = axs[1, 0].imshow(contour_img)
    axs[1, 0].set_title(f'3. Bordes y contornos\n(umbrales: {threshold1_init}, {threshold2_init})')
    axs[1, 0].axis('off')

    # 4. Bordes y contornos con ruido
    seed_init = 42
    def update_noise(seed):
        np.random.seed(int(seed))
        ruido = np.random.normal(0, 20, gray.shape).astype(np.uint8)
        gray_ruido = cv2.add(gray, ruido)
        canny_seed = cv2.Canny(gray_ruido, 150, 300)
        contornos_ruido, _ = cv2.findContours(canny_seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img_ruido = cv2.cvtColor(gray_ruido, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(contour_img_ruido, contornos_ruido, -1, (255, 0, 0), 2)
        axs[1, 1].imshow(contour_img_ruido)
        axs[1, 1].set_title(f'4. Canny + ruido (semilla={int(seed)})')
        axs[1, 1].axis('off')
        fig.canvas.draw_idle()
    update_noise(seed_init)

    axcolor = 'lightgoldenrodyellow'
    axthresh1 = plt.axes([0.25, 0.13, 0.5, 0.03], facecolor=axcolor)
    axthresh2 = plt.axes([0.25, 0.09, 0.5, 0.03], facecolor=axcolor)
    axseed = plt.axes([0.25, 0.05, 0.5, 0.03], facecolor=axcolor)
    slider_thresh1 = Slider(axthresh1, 'Umbral 1 (min)', 0, 255, valinit=threshold1_init, valstep=1)
    slider_thresh2 = Slider(axthresh2, 'Umbral 2 (max)', 0, 255, valinit=threshold2_init, valstep=1)
    slider_seed = Slider(axseed, 'Semilla ruido', 0, 100, valinit=seed_init, valstep=1)

    # Actualización solo para la imagen sin ruido (figura 3)
    slider_thresh1.on_changed(lambda val: update(val, slider_thresh1, slider_thresh2, gray, axs, im_edges, fig))
    slider_thresh2.on_changed(lambda val: update(val, slider_thresh1, slider_thresh2, gray, axs, im_edges, fig))
    slider_seed.on_changed(update_noise)

    plt.show()


# Función para actualizar la imagen de bordes Canny
def update(val, slider_thresh1, slider_thresh2, gray, axs, im_edges, fig):
    t1 = int(slider_thresh1.val)
    t2 = int(slider_thresh2.val)
    # Evitar que threshold1 sea mayor o igual a threshold2
    if t1 >= t2:
        axs[1, 0].set_title('¡Umbral 1 debe ser menor que Umbral 2!')
        im_edges.set_data(np.zeros_like(gray))
    else:
        edges_new = cv2.Canny(gray, t1, t2)
        # Encontrar contornos en la imagen de bordes
        contours, _ = cv2.findContours(edges_new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Dibujar los contornos sobre una imagen RGB para visualización
        contour_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 2)
    im_edges.set_data(contour_img)
    axs[1, 0].set_title(f'3. Bordes y contornos\n(umbrales: {t1}, {t2})')
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
