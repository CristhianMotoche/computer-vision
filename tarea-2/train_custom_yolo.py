# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "opencv-python",
#     "ultralytics",
#     "numpy",
#     "matplotlib",
# ]
# ///

import os
from pathlib import Path
from ultralytics import YOLO
import yaml


def setup_dataset_paths():
    """Corrige las rutas del dataset para que sean absolutas"""
    
    # Obtener ruta absoluta al directorio de datos
    data_dir = Path(__file__).parent / "data"
    dataset_config = data_dir / "data.yaml"
    
    if not dataset_config.exists():
        print(f"❌ Configuración del dataset no encontrada: {dataset_config}")
        return None
    
    # Leer configuración actual
    with open(dataset_config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Actualizar rutas para que sean absolutas
    base_path = data_dir.parent  # Subir un nivel desde data/
    
    # Crear configuración corregida
    corrected_config = {
        'path': str(base_path),  # Ruta raíz
        'train': 'data/train/images',  # Relativa a path
        'val': 'data/valid/images' if (base_path / 'data/valid').exists() else 'data/train/images',
        'test': 'data/test/images' if (base_path / 'data/test').exists() else 'data/train/images',
        'nc': config['nc'],
        'names': config['names']
    }
    
    # Guardar configuración corregida
    corrected_file = data_dir / "data_corrected.yaml"
    with open(corrected_file, 'w') as f:
        yaml.dump(corrected_config, f, default_flow_style=False)
    
    print(f"✓ Configuración de dataset corregida creada: {corrected_file}")
    print(f"✓ Clases: {corrected_config['names']}")
    print(f"✓ Número de clases: {corrected_config['nc']}")
    
    return str(corrected_file)


def train_custom_yolo():
    """Entrena YOLOv12 con dataset personalizado de gorras/sombreros"""
    
    print("=== Entrenamiento Personalizado YOLOv12: Detección de Gorras/Sombreros ===")
    print("Ajustando YOLOv12 nano con dataset personalizado\n")
    
    # Configurar dataset
    dataset_config = setup_dataset_paths()
    if not dataset_config:
        return
    
    # Cargar modelo pre-entrenado YOLOv12 nano
    print("Cargando modelo pre-entrenado YOLOv12 nano...")
    model = YOLO("yolo12n.pt")  # Cargar pesos pre-entrenados para transfer learning
    
    # Configurar parámetros de entrenamiento para CPU
    training_config = {
        'data': dataset_config,       # Configuración del dataset
        'epochs': 50,                 # Épocas reducidas para entrenamiento rápido
        'imgsz': 640,                # Tamaño de imagen
        'device': 'cpu',             # Usar CPU
        'batch': 4,                  # Tamaño de batch pequeño para CPU
        'patience': 10,              # Paciencia para parada temprana
        'save_period': 5,            # Guardar checkpoint cada 5 épocas
        'workers': 2,                # Número de workers para cargar datos
        'project': 'custom_training', # Directorio de salida
        'name': 'cap_hat_yolo12',    # Nombre del experimento
        'exist_ok': True,            # Sobrescribir resultados existentes
        'pretrained': True,          # Usar pesos pre-entrenados
        'verbose': True              # Mostrar progreso de entrenamiento
    }
    
    print(f"✓ Configuración de entrenamiento:")
    for key, value in training_config.items():
        print(f"   {key}: {value}")
    print()
    
    # Iniciar entrenamiento
    print("🚀 Iniciando ajuste fino...")
    print("Esto entrenará el modelo para detectar gorras y sombreros además de objetos COCO")
    print("El entrenamiento puede tomar varios minutos en CPU...\n")
    
    try:
        # Entrenar el modelo
        results = model.train(**training_config)
        
        print("\n✅ ¡Entrenamiento completado!")
        print(f"✓ Mejor modelo guardado en: custom_training/cap_hat_yolo12/weights/best.pt")
        print(f"✓ Último modelo guardado en: custom_training/cap_hat_yolo12/weights/last.pt")
        
        # Cargar el mejor modelo entrenado para pruebas
        best_model_path = "custom_training/cap_hat_yolo12/weights/best.pt"
        if os.path.exists(best_model_path):
            print(f"\n📊 Modelo listo para inferencia:")
            print(f"   Cargar con: model = YOLO('{best_model_path}')")
            
            # Prueba rápida
            trained_model = YOLO(best_model_path)
            print(f"   Clases: {trained_model.names}")
        
        return best_model_path
        
    except Exception as e:
        print(f"❌ Entrenamiento falló: {e}")
        return None


def test_custom_model(model_path):
    """Prueba el modelo entrenado con la cámara web"""
    
    if not os.path.exists(model_path):
        print(f"❌ Modelo entrenado no encontrado: {model_path}")
        return
    
    print(f"\n=== Probando Modelo Personalizado ===")
    print("Cargando modelo personalizado entrenado...")
    
    import cv2
    import time
    
    # Cargar modelo personalizado
    model = YOLO(model_path)
    
    print(f"✓ Modelo personalizado cargado")
    print(f"✓ Clases del modelo: {list(model.names.values())}")
    
    # Inicializar cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se puede acceder a la cámara")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✓ Cámara inicializada")
    print("✓ Presiona 'q' para salir")
    
    cv2.namedWindow("YOLOv12 Personalizado - Detección de Gorras/Sombreros", cv2.WINDOW_AUTOSIZE)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Ejecutar detección
            results = model(frame, conf=0.3, verbose=False)  # Confianza menor para objetos personalizados
            
            # Dibujar detecciones
            annotated_frame = results[0].plot()
            
            # Agregar información
            num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
            cv2.putText(annotated_frame, f"YOLOv12 Personalizado | Detecciones: {num_detections}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("YOLOv12 Personalizado - Detección de Gorras/Sombreros", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Prueba completada")


def main():
    """Función principal con opciones"""
    
    print("Elige una opción:")
    print("1. Entrenar modelo personalizado (ajustar YOLOv12 para detección de gorras/sombreros)")
    print("2. Probar modelo personalizado existente")
    print("3. Ambos (entrenar y luego probar)")
    
    choice = input("\nIngresa tu opción (1-3): ").strip()
    
    if choice == "1":
        train_custom_yolo()
    elif choice == "2":
        model_path = input("Ingresa la ruta del modelo (o presiona Enter para usar la predeterminada): ").strip()
        if not model_path:
            model_path = "custom_training/cap_hat_yolo12/weights/best.pt"
        test_custom_model(model_path)
    elif choice == "3":
        model_path = train_custom_yolo()
        if model_path:
            test_custom_model(model_path)
    else:
        print("Opción inválida")


if __name__ == "__main__":
    main()
