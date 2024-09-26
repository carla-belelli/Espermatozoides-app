import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import tempfile
from nd2reader import ND2Reader
import streamlit as st
from utils import normalizar_y_convertir , quitar_fondo_estático



tiempo_total=[] 

def procesar_video(video_path, confidence, stframe, progress_text, max_dist_threshold, num_frames=None, device=None, quitar_fondo_video=None, tracker=None):
    # Abrir el video utilizando ND2Reader
    with ND2Reader(video_path) as nd2_reader:
        # Obtener la cantidad total de frames y las dimensiones del video
        frame_count = nd2_reader.metadata['num_frames']
        height, width = nd2_reader.metadata['height'], nd2_reader.metadata['width']
        fps = int(nd2_reader.frame_rate)  # Asumiendo que la tasa de frames es de 100 FPS
        tiempo_total = frame_count / fps  # Calcular el tiempo total en segundos

        # Si se especifica num_frames, ajustar el tiempo total
        if num_frames is not None:
            tiempo_total = num_frames / fps

        # Crear un archivo temporal para guardar el video procesado
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para el video
        out = cv2.VideoWriter(output_file.name, fourcc, fps, (width, height))

        # Inicializar estructuras para seguimiento y conteo
        track_history = defaultdict(lambda: [])
        track_initialized = {}
        sperm_counts = []  # Lista para contar el número de espermatozoides detectados por frame
        trajectory_data = []  # Lista para almacenar datos de trayectoria
        bbox_sizes = defaultdict(list)  # Tamaños de las bounding boxes por ID
        frame_number = 0  # Contador de frames procesados
        last_frame = None  # Frame anterior para el seguimiento

        # Leer todos los frames del video y normalizarlos
        frames_list = [frame for frame in nd2_reader]
        frames_nd2 = np.array([normalizar_y_convertir(frame) for frame in frames_list])

        # Si se indica que se debe quitar el fondo, aplicar el filtro
        if quitar_fondo_video:
            frames_nd2 = quitar_fondo_estático(frames_nd2, frame_count)

        # Crear una lista de frames a colorear para el procesamiento visual
        frames_a_colorear = list(frames_nd2)

        # Procesar cada frame del video
        for frame_number, frame in enumerate(frames_nd2):
            # Romper el bucle si se han procesado el número de frames especificado
            if num_frames is not None and frame_number >= num_frames:
                break

            # Actualizar el texto de progreso con el frame actual
            progress_text.text(f"Procesando frame: {frame_number + 1} / {min(frame_count, num_frames) if num_frames else frame_count}")

            # Detectar espermatozoides en el frame actual
            dets = detectar_espermatozoides(frame, confidence, device)

            # Actualizar el tracker con las detecciones del frame
            tracks = tracker.update(dets, frame)
            sperm_count = len(dets)  # Contar el número de espermatozoides detectados
            sperm_counts.append(sperm_count)  # Agregar el conteo a la lista

            # Advertencia si se detectan demasiados espermatozoides
            if sperm_count > 500:
                stframe.warning("ATENCIÓN: Demasiados espermatozoides detectados (más de 500).")

            # Seguir trayectorias de los espermatozoides detectados
            frame, track_history, bbox_sizes, trajectory_data, last_frame = seguir_trayectorias(
                tracks, frame, last_frame, track_history, track_initialized, max_dist_threshold, trajectory_data, bbox_sizes, frame_number
            )

            # Escribir el frame procesado en el archivo de salida
            out.write(frame)

        out.release()  # Cerrar el objeto de escritura de video

        # Leer el video procesado para enviar los bytes a la aplicación
        with open(output_file.name, "rb") as f:
            video_bytes = f.read()

        # Crear un diccionario con información relevante del video procesado
        video_info = {
            'Número de Frames': frame_count,
            'Número de Frames a procesar': min(frame_count, num_frames) if num_frames else frame_count,
            'Resolución': f'{width}x{height}',
            'FPS': fps,
            'Tiempo Total a analizar (s)': tiempo_total
        }
        
        # Eliminar el texto de progreso una vez que se termine el procesamiento
        progress_text.empty()

    # Retornar los datos procesados
    return video_bytes, sperm_counts, track_history, last_frame, output_file, tiempo_total, video_info, bbox_sizes, fps, trajectory_data, frames_a_colorear

@st.cache_data(show_spinner=False)
def cargar_modelo_yolo():
    # Cargar el modelo YOLO una sola vez y cachearlo
    yolo_model = YOLO(r'yolov9espermatozoides.pt')  # Ruta del modelo
    return yolo_model

def detectar_espermatozoides(frame, confidence, device):
    # Cargar el modelo YOLO para la detección
    yolo_model = cargar_modelo_yolo()
    # Realizar la detección usando YOLO
    results = yolo_model(frame, max_det=1000, conf=confidence, device=device, imgsz=640)
    
    # Extraer las detecciones
    dets = []
    for result in results:
        for detection in result.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = detection  # Obtener coordenadas y confianza de la detección
            dets.append([x1, y1, x2, y2, conf, int(cls)])  # Agregar detección a la lista
            
    return np.array(dets)  # Retornar detecciones como un arreglo NumPy

def seguir_trayectorias(tracks, frame, last_frame, track_history, track_initialized, max_dist_threshold, trajectory_data, bbox_sizes, frame_number):
    # Extraer coordenadas y atributos de las trayectorias
    xyxys = tracks[:, 0:4].astype('int')  # Coordenadas de la bounding box
    ids = tracks[:, 4].astype('int')  # IDs de los objetos rastreados
    confs = tracks[:, 5].round(decimals=2)  # Confianza de las detecciones
    clss = tracks[:, 6].astype('int')  # Clases de los objetos

    prev_centroids = {}  # Diccionario para guardar centroides previos
    last_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convertir frame a BGR para OpenCV

    # Procesar cada detección en el frame
    for xyxy, id, conf, cls in zip(xyxys, ids, confs, clss):
        # Dibujar la bounding box en el frame
        frame = cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (100, 125, 0))
        # Escribir el ID en el frame
        frame = cv2.putText(frame, f'id: {id}', (xyxy[0], xyxy[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=1)
        last_frame = cv2.putText(last_frame, f'id: {id}', (xyxy[0], xyxy[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), thickness=1)

        # Calcular el centroide de la bounding box
        centroides = encontrar_centroide(frame, (xyxy[0], xyxy[1], xyxy[2], xyxy[3]))
        track = track_history[id]  # Obtener el historial de seguimiento para el ID

        # Calcular el tamaño de la bounding box
        width1 = xyxy[2] - xyxy[0]
        height1 = xyxy[3] - xyxy[1]
        bbox_size = (width1, height1)

        # Guardar el tamaño de la bounding box en el diccionario
        bbox_sizes[id].append(bbox_size)

        # Verificar si el seguimiento está inicializado
        if id not in track_initialized:
            if len(centroides) == 1:  # Si se detecta un solo centroide
                track_initialized[id] = True  # Marcar como inicializado
                cx, cy = centroides[0]  # Obtener coordenadas del centroide
                track.append((cx, cy))  # Agregar al historial
                trajectory_data.append({'ID': id, 'FRAME': frame_number, 'X': cx, 'Y': cy})  # Guardar datos de trayectoria
                cv2.circle(frame, (cx, cy), 1, (0, 0, 255), -1)  # Dibujar el centroide en el frame
            prev_centroids[id] = centroides  # Guardar centroides previos
        else:
                    # Verificar si hay un historial de seguimiento disponible
            if len(track) > 0:
                # Obtener las coordenadas del último centroide registrado en la trayectoria
                prev_cx, prev_cy = track[-1]
                closest_centroid = None  # Inicializar la variable para el centroide más cercano
                min_dist = float('inf')  # Inicializar la distancia mínima a un valor infinito

                # Iterar a través de los centroides detectados
                for cx, cy in centroides:
                    # Calcular la distancia euclidiana entre el último centroide registrado y el actual
                    dist = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                    # Verificar si la distancia está dentro del umbral y es la mínima encontrada
                    if dist <= max_dist_threshold and dist < min_dist:
                        min_dist = dist  # Actualizar la distancia mínima
                        closest_centroid = (cx, cy)  # Guardar el centroide más cercano

                # Si se encontró un centroide cercano
                if closest_centroid is not None:
                    cx, cy = closest_centroid  # Asignar las coordenadas del centroide más cercano
                    track.append((cx, cy))  # Agregar el centroide a la trayectoria
                    # Agregar los datos de la trayectoria al registro
                    trajectory_data.append({'ID': id, 'FRAME': frame_number, 'X': cx, 'Y': cy})
                    # Dibujar un pequeño círculo en el centroide detectado en el frame
                    cv2.circle(frame, (cx, cy), 1, (0, 0, 255), -1)

                # Actualizar el registro de centroides anteriores para el ID actual
                prev_centroids[id] = centroides

        # Retornar el frame procesado, el historial de seguimiento, tamaños de las bounding boxes, 
        # datos de trayectoria y el último frame
    return frame, track_history, bbox_sizes, trajectory_data, last_frame


# Función para encontrar el centroide de la cabeza del espermatozoide en la bounding box
def encontrar_centroide(image, bounding_box):
    # Convertir la imagen a escala de grises para facilitar el procesamiento
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x1, y1, x2, y2 = bounding_box  # Extraer las coordenadas de la bounding box
    roi = gray[y1:y2, x1:x2]  # Definir la región de interés (ROI) basada en la bounding box
    # Aplicar un umbral para binarizar la imagen
    _, thresholded = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY)
    # Encontrar los contornos en la imagen binarizada
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroides = []  # Lista para almacenar los centroides detectados

    # Iterar a través de los contornos encontrados
    for contour in contours:
        M = cv2.moments(contour)  # Calcular los momentos del contorno
        # Verificar que el área del contorno no sea cero para evitar división por cero
        if M["m00"] != 0:
            # Calcular las coordenadas del centroide
            cx = int(M["m10"] / M["m00"]) + x1  # Centroide en X
            cy = int(M["m01"] / M["m00"]) + y1  # Centroide en Y
            centroides.append((cx, cy))  # Agregar el centroide a la lista

    return centroides  # Retornar la lista de centroides encontrados


tiempo_total=[] 
def procesar_video(video_path, confidence, stframe, progress_text, max_dist_threshold, num_frames=None, device=None, quitar_fondo=None, tracker=None):
        with ND2Reader(video_path) as nd2_reader:
            frame_count = nd2_reader.metadata['num_frames']
            height, width = nd2_reader.metadata['height'], nd2_reader.metadata['width']
            #fps = int(nd2_reader.frame_rate)  # Asumiendo 100 FPS
            fps= 100
            tiempo_total = frame_count / fps
            if num_frames is not None:
                tiempo_total = num_frames / fps

            output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file.name, fourcc, fps, (width, height))

            track_history = defaultdict(lambda: [])
            track_initialized = {}
            sperm_counts = []
            trajectory_data = []
            bbox_sizes = defaultdict(list)
            frame_number = 0
            last_frame = None

            frames_list = [frame for frame in nd2_reader]
            frames_nd2 = np.array([normalizar_y_convertir(frame) for frame in frames_list])

            if quitar_fondo:
                frames_nd2 = quitar_fondo_estático(frames_nd2, frame_count)

            frames_a_colorear = list(frames_nd2)

            for frame_number, frame in enumerate(frames_nd2):
                if num_frames is not None and frame_number >= num_frames:
                    break

                progress_text.text(f"Procesando frame: {frame_number + 1} / {min(frame_count, num_frames) if num_frames else frame_count}")

                # Detectar espermatozoides
                dets = detectar_espermatozoides(frame, confidence, device)

                # Actualizar el tracker con las detecciones
                tracks = tracker.update(dets, frame)
                sperm_count = len(dets)
                sperm_counts.append(sperm_count)

                if sperm_count > 500:
                    stframe.warning("ATENCIÓN: Demasiados espermatozoides detectados (más de 500).")

                # Seguir trayectorias
                frame, track_history, bbox_sizes, trajectory_data, last_frame = seguir_trayectorias(tracks, frame, last_frame, track_history, track_initialized, max_dist_threshold, trajectory_data, bbox_sizes, frame_number)

                # Escribir el frame procesado
                out.write(frame)

            out.release()

            with open(output_file.name, "rb") as f:
                video_bytes = f.read()

            video_info = {
                'Número de Frames': frame_count,
                'Número de Frames a procesar': min(frame_count, num_frames) if num_frames else frame_count,
                'Resolución': f'{width}x{height}',
                'FPS': fps,
                'Tiempo Total a analizar (s)': tiempo_total
            }
            # Eliminar el texto de progreso una vez que se termine el procesamiento
            progress_text.empty()

        return video_bytes, sperm_counts, track_history, last_frame, output_file, tiempo_total, video_info, bbox_sizes, fps, trajectory_data, frames_a_colorear


