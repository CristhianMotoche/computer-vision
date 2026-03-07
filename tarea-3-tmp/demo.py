#!/usr/bin/env python3
"""
Script de ejemplo para demostrar el uso del sistema FaceNet
Crea un dataset sintético y demuestra el entrenamiento y reconocimiento
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

def create_sample_dataset():
    """Crea un dataset de ejemplo para demostración"""
    dataset_dir = "dataset"
    
    print("Creando dataset de ejemplo...")
    
    # Datos de personas ficticias
    people = ["Juan", "Maria", "Carlos"]
    
    for person in people:
        person_dir = os.path.join(dataset_dir, person)
        os.makedirs(person_dir, exist_ok=True)
        
        # Crear 3 imágenes sintéticas por persona
        for i in range(3):
            # Generar imagen sintética (en un caso real, estas serían fotos reales)
            img = generate_synthetic_face(person, i)
            img_path = os.path.join(person_dir, f"{person}_{i+1}.jpg")
            cv2.imwrite(img_path, img)
        
        print(f"Creadas imágenes para {person}")
    
    print(f"Dataset actualizado en: {dataset_dir}")
    return dataset_dir

def generate_synthetic_face(person_name, variant):
    """Genera una imagen sintética de rostro para demostración"""
    # En un caso real, estas serían fotografías reales de personas
    # Aquí creamos imágenes sintéticas simples para demostración
    
    # Crear imagen base
    img = np.random.randint(50, 200, (160, 160, 3), dtype=np.uint8)
    
    # Añadir características distintivas basadas en el nombre
    person_hash = hash(person_name) % 256
    
    # Color base único para cada persona
    color_shift = person_hash % 100
    img[:, :, 0] = np.clip(img[:, :, 0] + color_shift, 0, 255)
    
    # Patrón único por variante
    variant_pattern = (variant + 1) * 30
    img[40:120, 40:120] = np.clip(img[40:120, 40:120] + variant_pattern, 0, 255)
    
    # Añadir algo de ruido
    noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img

def demo_training():
    """Demuestra el entrenamiento del modelo"""
    print("\n=== Demostración de Entrenamiento FaceNet ===")
    
    # Crear dataset sintético
    dataset_dir = create_sample_dataset()
    
    # Importar el trainer
    from train_facenet import FaceNetTrainer
    
    # Crear trainer
    trainer = FaceNetTrainer(
        data_dir=dataset_dir,
        model_save_path="demo_models/",
        batch_size=8  # Batch pequeño para demo
    )
    
    print("\nIniciando entrenamiento de demostración...")
    print("(Este es un ejemplo con datos sintéticos - usa datos reales para casos reales)")
    
    # Entrenar por pocas épocas para demostración
    trainer.train(
        epochs=5,  # Pocas épocas para demo rápida
        steps_per_epoch=10,
        validation_steps=3
    )
    
    return "demo_models/facenet_final.h5", dataset_dir

def demo_recognition(model_path, dataset_dir):
    """Demuestra el reconocimiento facial"""
    print("\n=== Demostración de Reconocimiento Facial ===")
    
    # Importar el reconocedor
    from face_recognizer import FaceRecognizer
    
    # Crear reconocedor
    recognizer = FaceRecognizer(model_path)
    
    # Construir base de datos
    print("Construyendo base de datos de rostros...")
    recognizer.build_database_from_directory(dataset_dir)
    
    # Guardar base de datos
    db_path = "demo_face_database.pkl"
    recognizer.save_database(db_path)
    
    # Probar reconocimiento con algunas imágenes
    print("\nProbando reconocimiento...")
    
    test_images = []
    for person_dir in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person_dir)
        if os.path.isdir(person_path):
            for img_file in os.listdir(person_path)[:2]:  # Tomar 2 imágenes por persona
                if img_file.endswith('.jpg'):
                    test_images.append((os.path.join(person_path, img_file), person_dir))
    
    print(f"\nProbando con {len(test_images)} imágenes:")
    
    correct = 0
    total = 0
    
    for img_path, true_person in test_images:
        predicted, distance, confidence = recognizer.identify_person(img_path, threshold=1.5)
        total += 1
        
        if predicted == true_person:
            correct += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"{status} {os.path.basename(img_path)}: {predicted} "
              f"(verdad: {true_person}, dist: {distance:.3f})")
    
    accuracy = correct / total if total > 0 else 0
    print(f"\nPrecisión en demo: {accuracy:.1%} ({correct}/{total})")
    
    return recognizer

def demo_face_comparison(recognizer, dataset_dir):
    """Demuestra comparación directa de rostros"""
    print("\n=== Demostración de Comparación de Rostros ===")
    
    # Obtener algunas imágenes para comparar
    image_files = []
    for person_dir in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person_dir)
        if os.path.isdir(person_path):
            for img_file in os.listdir(person_path)[:1]:  # Una imagen por persona
                if img_file.endswith('.jpg'):
                    image_files.append(os.path.join(person_path, img_file))
    
    if len(image_files) >= 2:
        # Comparar misma persona (primeras dos imágenes de la misma carpeta)
        person_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        if person_dirs:
            first_person_dir = os.path.join(dataset_dir, person_dirs[0])
            person_images = [f for f in os.listdir(first_person_dir) if f.endswith('.jpg')]
            
            if len(person_images) >= 2:
                img1 = os.path.join(first_person_dir, person_images[0])
                img2 = os.path.join(first_person_dir, person_images[1])
                
                distance, similarity = recognizer.compare_two_faces(img1, img2)
                print(f"Misma persona - Distancia: {distance:.4f}, Similitud: {similarity:.4f}")
        
        # Comparar personas diferentes
        if len(image_files) >= 2:
            distance, similarity = recognizer.compare_two_faces(image_files[0], image_files[1])
            print(f"Personas diferentes - Distancia: {distance:.4f}, Similitud: {similarity:.4f}")

def main():
    """Función principal de demostración"""
    print("=== Sistema FaceNet - Demostración Completa ===")
    print("Este script demuestra el entrenamiento y uso de FaceNet para reconocimiento facial")
    print("NOTA: Esta demo usa imágenes sintéticas. Para uso real, utiliza fotografías reales.")
    
    try:
        # 1. Demostrar entrenamiento
        model_path, dataset_dir = demo_training()
        
        # 2. Demostrar reconocimiento
        recognizer = demo_recognition(model_path, dataset_dir)
        
        # 3. Demostrar comparación
        demo_face_comparison(recognizer, dataset_dir)
        
        print("\n=== Demostración Completada ===")
        print("\nPara usar con datos reales:")
        print("1. Organiza tus imágenes en carpetas por persona")
        print("2. Ejecuta: python train_facenet.py --data_dir tu_dataset --epochs 100")
        print("3. Usa: python face_recognizer.py --mode build_db --data_dir tu_dataset --model_path models/facenet_final.h5")
        print("4. Identifica: python face_recognizer.py --mode identify --image1 nueva_imagen.jpg --model_path models/facenet_final.h5")
        
    except Exception as e:
        print(f"Error durante la demostración: {e}")
        print("Asegúrate de tener todas las dependencias instaladas:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()