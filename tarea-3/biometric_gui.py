import tkinter as tk
from tkinter import messagebox, ttk, simpledialog
import cv2
import os
import time
import numpy as np
from PIL import Image, ImageTk
import threading
from mtcnn import MTCNN
from keras_facenet import FaceNet
import subprocess
import platform


class BiometricGUI:
    # Configuration constants
    IMAGES_PER_POSE = 3
    
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema Biométrico - Grupo 8")
        # Set to full screen
        self.root.attributes('-fullscreen', True)
        # Allow escape to exit fullscreen
        self.root.bind('<Escape>', lambda e: self.root.attributes('-fullscreen', False))
        self.root.configure(bg="#f0f0f0")

        # Variables
        self.cap = None
        self.is_capturing = False
        self.is_authenticating = False
        self.current_user = None
        self.pose_index = 0
        self.img_count = 0
        self.poses = ["frontal", "izquierda", "derecha", "arriba", "abajo"]
        self.is_authenticated = False
        self.authenticated_user = None
        self.auth_timeout = 30  # seconds
        self.last_auth_time = 0

        # Modelos
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.detector = MTCNN()
        self.embedder = FaceNet()

        self.setup_ui()

    def setup_ui(self):
        # Título principal
        title_label = tk.Label(
            self.root,
            text="Sistema Biométrico",
            font=("Arial", 24, "bold"),
            bg="#f0f0f0",
            fg="#2c3e50",
        )
        title_label.pack(pady=20)

        # Frame principal para botones
        button_frame = tk.Frame(self.root, bg="#f0f0f0")
        button_frame.pack(pady=20)  # Reduced from 50

        # Botón para agregar nuevo usuario
        add_user_btn = tk.Button(
            button_frame,
            text="1. Agregar Nuevo Usuario",
            font=("Arial", 16, "bold"),
            bg="#3498db",
            fg="white",
            width=25,
            height=2,
            command=self.add_new_user,
        )
        add_user_btn.pack(pady=10)  # Reduced from 20

        # Botón para autenticación biométrica (toggle functionality)
        self.auth_btn = tk.Button(
            button_frame,
            text="2. Autenticación Biométrica",
            font=("Arial", 16, "bold"),
            bg="#e74c3c",
            fg="white",
            width=25,
            height=2,
            command=self.toggle_biometric_auth,
        )
        self.auth_btn.pack(pady=10)

        # Botón para archivo seguro (deshabilitado por defecto)
        self.secure_file_btn = tk.Button(
            button_frame,
            text="3. Abrir Archivo Seguro",
            font=("Arial", 16, "bold"),
            bg="#95a5a6",  # Gray color when disabled
            fg="white",
            width=25,
            height=2,
            command=self.open_secure_file,
            state="disabled"
        )
        self.secure_file_btn.pack(pady=10)  # Reduced from 20

        # Botón de logout (deshabilitado por defecto)
        self.logout_btn = tk.Button(
            button_frame,
            text="4. Cerrar Sesión",
            font=("Arial", 14, "bold"),
            bg="#95a5a6",  # Gray color when disabled
            fg="white",
            width=20,
            height=1,
            command=self.logout,
            state="disabled"
        )
        self.logout_btn.pack(pady=5)  # Reduced from 10

        # Label de estado de autenticación
        self.auth_status_label = tk.Label(
            self.root,
            text="Estado: No autenticado",
            font=("Arial", 12, "bold"),
            bg="#f0f0f0",
            fg="#e74c3c"
        )
        self.auth_status_label.pack(pady=5)  # Reduced from 10

        # Frame para video
        self.video_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.video_frame.pack(pady=10)  # Reduced from 20

        # Label para video
        self.video_label = tk.Label(self.video_frame, bg="#f0f0f0")
        self.video_label.pack()

        # Frame para controles
        self.control_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.control_frame.pack(pady=5)  # Reduced from 10

        # Label para instrucciones
        self.instruction_label = tk.Label(
            self.root, text="", font=("Arial", 14), bg="#f0f0f0", fg="#2c3e50"
        )
        self.instruction_label.pack(pady=5)  # Reduced from 10

    def add_new_user(self):
        """Función para agregar nuevo usuario"""
        if self.is_capturing or self.is_authenticating:
            return

        # Pedir nombre del usuario
        user_name = simpledialog.askstring(
            "Nuevo Usuario", "Ingrese el nombre del usuario:"
        )
        if not user_name:
            return

        self.current_user = user_name.strip().lower().replace(" ", "_")

        # Crear directorio para el usuario
        user_dir = f"dataset/{self.current_user}"
        os.makedirs(user_dir, exist_ok=True)

        # Resetear variables
        self.pose_index = 0
        self.img_count = 0

        # Iniciar captura
        self.start_capture()

    def start_capture(self):
        """Iniciar captura de rostros"""
        self.is_capturing = True
        self.cap = cv2.VideoCapture(0)

        # Mostrar el frame de video si estaba oculto
        self.video_frame.pack(pady=10)

        # Iniciar thread de captura
        self.capture_thread = threading.Thread(target=self.capture_faces)
        self.capture_thread.start()

    def capture_faces(self):
        """Capturar rostros para entrenamiento"""
        last_capture = time.time()

        while self.is_capturing and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Detectar rostros
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5
            )

            if len(faces) > 0:
                (x, y, w, h) = faces[0]

                # Dibujar bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Instrucción actual
                instruction = f"Mire hacia: {self.poses[self.pose_index]} ({self.img_count % self.IMAGES_PER_POSE + 1}/{self.IMAGES_PER_POSE})"
                cv2.putText(
                    frame,
                    instruction,
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

                # Actualizar instrucción en GUI
                self.root.after(
                    0, lambda: self.instruction_label.config(text=instruction)
                )

                # Capturar imagen cada 2 segundos
                if time.time() - last_capture > 2:
                    user_dir = f"dataset/{self.current_user}"
                    img_name = f"{self.poses[self.pose_index]}_{self.img_count}.jpg"
                    cv2.imwrite(os.path.join(user_dir, img_name), frame)

                    self.img_count += 1
                    last_capture = time.time()

                    # Cambiar pose después de las imágenes configuradas
                    if self.img_count % self.IMAGES_PER_POSE == 0:
                        self.pose_index += 1
                        if self.pose_index >= len(self.poses):
                            # Completado - procesar embeddings
                            self.finish_capture()
                            break

            # Mostrar video en GUI
            self.display_frame(frame)

    def finish_capture(self):
        """Finalizar captura y generar embeddings"""
        self.stop_capture()

        # Ocultar el frame de video
        self.root.after(0, lambda: self.video_frame.pack_forget())

        # Mostrar mensaje de procesamiento
        self.root.after(
            0,
            lambda: self.instruction_label.config(
                text="Guardando embeddings LBPH y FaceNet... Por favor espere."
            ),
        )

        # Procesar en thread separado
        process_thread = threading.Thread(target=self.process_embeddings)
        process_thread.start()

    def process_embeddings(self):
        """Procesar embeddings LBPH y FaceNet"""
        try:
            # Generar modelo LBPH
            self.generate_lbph_model()

            # Generar embeddings FaceNet
            self.generate_facenet_embeddings()

            # Mostrar mensaje de éxito
            self.root.after(
                0,
                lambda: self.instruction_label.config(
                    text="¡Embeddings guardados exitosamente!"
                ),
            )
            # Pausa breve antes del mensaje final
            time.sleep(1)
            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    "Éxito", f"Usuario '{self.current_user}' registrado exitosamente!"
                ),
            )

        except Exception as e:
            self.root.after(
                0, lambda: messagebox.showerror("Error", f"Error al procesar: {str(e)}")
            )
        finally:
            self.root.after(0, lambda: self.instruction_label.config(text=""))

    def generate_lbph_model(self):
        """Generar modelo LBPH"""
        dataset_dir = f"dataset/{self.current_user}"
        model_path = f"models/{self.current_user}_lbph_model.yml"

        os.makedirs("models", exist_ok=True)

        faces = []
        labels = []

        for file in os.listdir(dataset_dir):
            if not file.lower().endswith(".jpg"):
                continue

            img_path = os.path.join(dataset_dir, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, (224, 224))
            img = cv2.equalizeHist(img)

            faces.append(img)
            labels.append(0)  # Un solo usuario por modelo

        if len(faces) == 0:
            raise ValueError("No se encontraron imágenes válidas")

        faces = np.array(faces)
        labels = np.array(labels)

        model = cv2.face.LBPHFaceRecognizer_create(
            radius=2, neighbors=16, grid_x=8, grid_y=8
        )
        model.train(faces, labels)
        model.save(model_path)

    def generate_facenet_embeddings(self):
        """Generar embeddings FaceNet"""
        dataset_dir = f"dataset/{self.current_user}"
        output_path = f"models/{self.current_user}_facenet.npy"

        embeddings = []

        for img_name in os.listdir(dataset_dir):
            img_path = os.path.join(dataset_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.detector.detect_faces(rgb)

            if len(faces) == 0:
                continue

            x, y, w, h = faces[0]["box"]
            face = rgb[y : y + h, x : x + w]

            face = cv2.resize(face, (160, 160))
            face = np.expand_dims(face, axis=0)

            embedding = self.embedder.embeddings(face)[0]
            embeddings.append(embedding)

        if len(embeddings) == 0:
            raise ValueError("No se pudieron generar embeddings")

        embeddings = np.array(embeddings)
        np.save(output_path, embeddings)

    def toggle_biometric_auth(self):
        """Toggle entre iniciar y cancelar autenticación biométrica"""
        if self.is_authenticating:
            # Si está autenticando, cancelar
            self.stop_auth()
        else:
            # Si no está autenticando, iniciar
            self.biometric_auth()

    def biometric_auth(self):
        """Función para autenticación biométrica"""
        if self.is_capturing:  # Don't start auth if capturing
            return

        # Verificar si existen modelos
        if not self.check_models_exist():
            messagebox.showerror(
                "Error",
                "No se encontraron usuarios registrados. Por favor agregue un usuario primero.",
            )
            return

        self.is_authenticating = True
        
        # Cambiar el botón a modo cancelar
        self.auth_btn.config(
            text="2. Cancelar Autenticación",
            bg="#f39c12"  # Orange color for cancel
        )
        
        self.cap = cv2.VideoCapture(0)

        # Mostrar el frame de video si estaba oculto
        self.video_frame.pack(pady=10)

        # Cargar modelos
        self.load_auth_models()

        # Iniciar thread de autenticación
        self.auth_thread = threading.Thread(target=self.authenticate)
        self.auth_thread.start()

    def check_models_exist(self):
        """Verificar si existen modelos entrenados"""
        models_dir = "models"
        if not os.path.exists(models_dir):
            return False

        # Buscar cualquier modelo LBPH o FaceNet
        for file in os.listdir(models_dir):
            if file.endswith("_lbph_model.yml") or file.endswith("_facenet.npy"):
                return True
        return False

    def load_auth_models(self):
        """Cargar modelos para autenticación"""
        models_dir = "models"
        self.auth_models = []

        # Buscar todos los modelos disponibles
        lbph_models = {}
        facenet_models = {}

        for file in os.listdir(models_dir):
            if file.endswith("_lbph_model.yml"):
                user = file.replace("_lbph_model.yml", "")
                lbph_models[user] = os.path.join(models_dir, file)
            elif file.endswith("_facenet.npy"):
                user = file.replace("_facenet.npy", "")
                facenet_models[user] = os.path.join(models_dir, file)

        # Cargar modelos para usuarios que tengan ambos tipos
        for user in set(lbph_models.keys()) & set(facenet_models.keys()):
            try:
                # Cargar LBPH
                lbph_model = cv2.face.LBPHFaceRecognizer_create()
                lbph_model.read(lbph_models[user])

                # Cargar FaceNet embeddings
                embeddings = np.load(facenet_models[user])

                self.auth_models.append(
                    {"user": user, "lbph": lbph_model, "embeddings": embeddings}
                )
            except:
                continue

    def authenticate(self):
        """Realizar autenticación en tiempo real"""
        frame_count = 0
        last_auth_result = None
        last_auth_time = 0
        
        while self.is_authenticating and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = time.time()
            
            # Use fast Haar cascade for initial detection every frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_haar = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60)
            )

            # Only run expensive authentication every 10 frames (about 3 times per second)
            # or if 2 seconds have passed since last authentication
            run_auth = (frame_count % 10 == 0) or (current_time - last_auth_time > 2.0)
            
            if len(faces_haar) > 0 and run_auth:
                # Use MTCNN only when we need to authenticate
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces_mtcnn = self.detector.detect_faces(rgb)
                
                if len(faces_mtcnn) > 0:
                    face_data = faces_mtcnn[0]  # Use first detected face
                    x, y, w, h = face_data["box"]

                    # Preparar rostro para LBPH
                    face_gray = cv2.cvtColor(
                        frame[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY
                    )
                    face_gray = cv2.resize(face_gray, (224, 224))
                    face_gray = cv2.equalizeHist(face_gray)

                    # Preparar rostro para FaceNet
                    face_rgb = rgb[y : y + h, x : x + w]
                    face_rgb = cv2.resize(face_rgb, (160, 160))
                    face_emb = self.embedder.embeddings(np.expand_dims(face_rgb, axis=0))[0]

                    # Probar con todos los modelos
                    best_match = None
                    best_score = 0

                    for model_data in self.auth_models:
                        # Test LBPH
                        label, confidence = model_data["lbph"].predict(face_gray)
                        lbph_score = max(0, min(100, 100 - confidence))

                        # Test FaceNet
                        similarities = [
                            self.cosine_similarity(face_emb, emb)
                            for emb in model_data["embeddings"]
                        ]
                        facenet_score = max(similarities) * 100

                        # Usar el mejor score
                        combined_score = max(lbph_score, facenet_score)

                        if (
                            combined_score > best_score and combined_score > 60
                        ):  # Umbral mínimo
                            best_score = combined_score
                            best_match = model_data["user"]

                    # Store authentication result
                    if best_match:
                        last_auth_result = {
                            "match": best_match, 
                            "score": best_score,
                            "color": (0, 255, 0),
                            "text": f"ACCESO: {best_match} ({best_score:.1f}%)"
                        }
                        # Enable secure file access
                        if not self.is_authenticated:
                            self.is_authenticated = True
                            self.authenticated_user = best_match
                            self.last_auth_time = current_time
                            self.root.after(0, self.enable_secure_access)
                        else:
                            # Update authentication time
                            self.last_auth_time = current_time
                    else:
                        last_auth_result = {
                            "match": None,
                            "score": 0,
                            "color": (0, 0, 255),
                            "text": "ACCESO DENEGADO"
                        }
                        # Disable secure file access if previously authenticated
                        if self.is_authenticated:
                            self.is_authenticated = False
                            self.authenticated_user = None
                            self.root.after(0, self.disable_secure_access)
                    
                    last_auth_time = current_time

            # Draw results on all detected faces using Haar cascade (fast)
            for (x, y, w, h) in faces_haar:
                if last_auth_result:
                    color = last_auth_result["color"]
                    text = last_auth_result["text"]
                else:
                    color = (255, 255, 0)  # Yellow for processing
                    text = "PROCESANDO..."

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                )

            # Update instruction label only when status changes
            current_status = ""
            if last_auth_result:
                current_status = last_auth_result["text"]
            elif len(faces_haar) > 0:
                current_status = "PROCESANDO..."
            else:
                current_status = "Esperando rostro..."
                
            if not hasattr(self, '_last_status') or self._last_status != current_status:
                self._last_status = current_status
                self.root.after(0, lambda s=current_status: self.instruction_label.config(text=s))

            # Check for authentication timeout
            if self.is_authenticated and (current_time - self.last_auth_time > self.auth_timeout):
                self.is_authenticated = False
                self.authenticated_user = None
                self.root.after(0, self.disable_secure_access)

            # Only update display every 3 frames for smoother performance
            if frame_count % 3 == 0:
                self.display_frame(frame)

    def cosine_similarity(self, a, b):
        """Calcular similitud coseno"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def display_frame(self, frame):
        """Mostrar frame en la GUI - optimized version"""
        # Redimensionar frame solo una vez
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convertir a formato PIL y ImageTk en una sola operación
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        # Actualizar label en una sola llamada
        def update_image():
            self.video_label.configure(image=imgtk)
            self.video_label.image = imgtk  # Keep a reference
            
        self.root.after(0, update_image)

    def stop_capture(self):
        """Detener captura"""
        self.is_capturing = False
        if self.cap:
            self.cap.release()

        # Ocultar el frame de video
        self.video_frame.pack_forget()

        # Limpiar controles
        for widget in self.control_frame.winfo_children():
            widget.destroy()

    def stop_auth(self):
        """Detener autenticación"""
        self.is_authenticating = False
        if self.cap:
            self.cap.release()

        # Ocultar el frame de video
        self.video_frame.pack_forget()

        # Restaurar el botón a su estado original
        self.auth_btn.config(
            text="2. Autenticación Biométrica",
            bg="#e74c3c"  # Red color for start
        )

        # Limpiar controles (backup for any remaining controls)
        for widget in self.control_frame.winfo_children():
            widget.destroy()

        self.instruction_label.config(text="")

    def enable_secure_access(self):
        """Habilitar acceso al archivo seguro"""
        self.secure_file_btn.config(
            state="normal",
            bg="#27ae60",  # Green color when enabled
            text=f"3. Archivo Seguro (Autorizado: {self.authenticated_user})"
        )
        self.logout_btn.config(
            state="normal",
            bg="#e67e22"  # Orange color for logout
        )
        self.auth_status_label.config(
            text=f"Estado: Autenticado como {self.authenticated_user}",
            fg="#27ae60"
        )

    def disable_secure_access(self):
        """Deshabilitar acceso al archivo seguro"""
        self.secure_file_btn.config(
            state="disabled",
            bg="#95a5a6",  # Gray color when disabled
            text="3. Abrir Archivo Seguro"
        )
        self.logout_btn.config(
            state="disabled",
            bg="#95a5a6"  # Gray color when disabled
        )
        self.auth_status_label.config(
            text="Estado: No autenticado",
            fg="#e74c3c"
        )

    def logout(self):
        """Cerrar sesión manualmente"""
        if self.is_authenticated:
            result = messagebox.askyesno(
                "Cerrar Sesión",
                f"¿Está seguro de que desea cerrar la sesión de {self.authenticated_user}?"
            )
            if result:
                self.is_authenticated = False
                self.authenticated_user = None
                self.disable_secure_access()
                messagebox.showinfo("Sesión Cerrada", "Ha cerrado sesión exitosamente.")

    def open_secure_file(self):
        """Abrir archivo seguro solo si está autenticado"""
        if not self.is_authenticated:
            messagebox.showerror("Acceso Denegado", "Debe autenticarse primero para acceder al archivo seguro.")
            return

        try:
            # Update access timestamp in the file
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            secure_file_path = "secure_file.txt"
            
            # Read current content
            with open(secure_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Update the timestamp in the content
            content = content.replace(
                "Fecha de último acceso: [Se actualiza automáticamente]",
                f"Fecha de último acceso: {timestamp} - Usuario: {self.authenticated_user}"
            )
            
            # Write back with timestamp
            with open(secure_file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # Open the file with the default text editor
            if platform.system() == "Windows":
                os.startfile(secure_file_path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", secure_file_path])
            else:  # Linux and other Unix-like systems
                subprocess.run(["xdg-open", secure_file_path])

            messagebox.showinfo(
                "Acceso Concedido", 
                f"Archivo seguro abierto exitosamente.\nUsuario autenticado: {self.authenticated_user}\nFecha de acceso: {timestamp}"
            )

        except Exception as e:
            messagebox.showerror("Error", f"Error al abrir el archivo seguro: {str(e)}")


if __name__ == "__main__":
    # Silenciar logs de TensorFlow
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    root = tk.Tk()
    app = BiometricGUI(root)
    root.protocol(
        "WM_DELETE_WINDOW",
        lambda: (
            app.stop_capture() if app.is_capturing else None,
            app.stop_auth() if app.is_authenticating else None,
            root.destroy(),
        ),
    )
    root.mainloop()
