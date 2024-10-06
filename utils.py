import streamlit as st
import cv2
import numpy as np
import os
from moviepy.editor import VideoFileClip

# Función para mostrar las instrucciones de uso
def mostrar_instrucciones():
    st.title("Bienvenido a la Herramienta de Detección y Análisis de Motilidad Espermática")
    st.write("""
        Esta aplicación permite procesar videos para detectar y analizar trayectorias de espermatozoides.

        ### Instrucciones de Uso
        1. **Cargar Videos**: Utiliza el botón "Sube un video" en la barra lateral para seleccionar uno o varios archivos de video en formato ND2.
        2. **Verificar Disponibilidad de GPU**: Activa la opción "¿Verificar disponibilidad de GPU?" si deseas utilizar la GPU para el procesamiento. Si no hay GPU disponible, se usará la CPU automáticamente.
        3. **Configuración de Parámetros**: Ingresa los siguientes parámetros:
           - **Confianza (Confidence)**: Umbral de confianza para la detección (valor entre 0.0 y 1.0, se recomienda comenzar con un valor de 0.1).
           - **Número de Frames (Num Frames)**: Total de frames a procesar (se recomienda un número que represente una buena muestra del video).
           - **Número Mínimo de Puntos en Trayectorias (Min Points)**: Cantidad mínima de puntos para considerar una trayectoria válida.
           - **Conversión de Píxeles a Micrómetros**: Factor para convertir distancias de píxeles a micrómetros (por defecto: 1.30).
           - **Distancia Umbral para Trayectorias**: Define el umbral para verificar que la distancia entre las coordenadas (x, y) de los puntos en la trayectoria no sea excesiva. Si la distancia es muy grande, podría indicar que se está tomando la coordenada de otro espermatozoide.
           - **Seleccione el Tracker**: Escoge el método de seguimiento que deseas utilizar (OCSORT o  BoTSORT).
        4. **Opciones Adicionales**:
           - **Quitar Fondo Estático**: Activa esta opción si deseas eliminar el fondo estático del video.
        5. **Procesar Videos**: Haz clic en "Procesar" para iniciar el análisis. Espera a que se complete el procesamiento. Si se detectan más de 500 espermatozoides, recibirás una advertencia, ya que muchas detecciones pueden afectar el rendimiento y la precisión del análisis.
        6. **Resultados**: Una vez procesado el video, se mostrarán las detecciones y el análisis de las trayectorias. Además, tendrás la opción de descargar los resultados.

        Si tienes dudas, consulta la documentación o los comentarios en el código.
    """)

def quitar_fondo_estático(frames_nd2, frame_count):
    """
    Aplica el filtro de mediana a un conjunto de frames ND2 para eliminar el fondo estático.
    """
    # Suponiendo que frames_nd2 tiene el formato (num_frames, height, width, channels)
    frame_shape = frames_nd2[0].shape  # Obtener la forma del primer frame
    y, x, channels = frame_shape  # Descomponer la forma en altura, ancho y canales
    progress_text = st.empty()  # Espacio vacío para mostrar el progreso
    stframe = st.empty()  # Espacio vacío para mostrar los frames procesados

    def fondo_mediana(frames_nd2, t):
        """
        Calcula el fondo como la mediana de `t` frames para cada píxel en cada canal.
        """
        # Inicializa la matriz para el fondo con la mediana
        processed_array = np.zeros((y, x, channels), dtype=frames_nd2.dtype)
        progress_text.text("Calculando fondo estático")  # Mensaje de progreso

        # Itera sobre cada canal
        for ch in range(channels):
            # Itera sobre cada píxel en el ancho
            for xx in range(x):
                # Itera sobre cada píxel en la altura
                for yy in range(y):
                    # Extrae los primeros `t` frames en una matriz temporal para el canal actual
                    temporal_frames = frames_nd2[:t, yy, xx, ch]
                    
                    # Calcula la mediana de los valores en la matriz temporal
                    median_value = np.median(temporal_frames)
                    
                    # Asigna la mediana al píxel correspondiente en el fondo para el canal actual
                    processed_array[yy, xx, ch] = median_value

        return processed_array  # Retorna el fondo calculado

    # Calcula el fondo en escala de grises
    fondo = fondo_mediana(frames_nd2, frame_count)
    #stframe.image(fondo)  # Muestra el fondo calculado
    #stframe.empty()  # Limpia el espacio de imagen

    # Calcula las diferencias entre el fondo y cada frame en escala de grises
    diferencias = []  # Lista para almacenar las diferencias
    for frame in frames_nd2:
        diff_frame = cv2.absdiff(fondo, frame)  # Calcula la diferencia entre el fondo y el frame
        
        # Ajustar brillo añadiendo un valor constante (beta)
        diff_frame_brillante = cv2.convertScaleAbs(diff_frame, alpha=1, beta=40)  # Ajustar beta para aumentar el brillo

        diferencias.append(diff_frame_brillante)  # Agrega el frame brillante a la lista
    
    # Convierte las diferencias en escala de grises de nuevo a RGB
    #for i, diff_frame_brillante in enumerate(diferencias):
        #stframe.image(diff_frame_brillante, caption=f"Frame {i+1}")  # Muestra cada frame procesado
        #stframe.empty()  # Limpia el espacio de imagen
    progress_text.empty()
    return diferencias  # Retorna la lista de frames con el fondo eliminado

# Función para normalizar y convertir el frame a uint8
def normalizar_y_convertir(frame):
    # Mostrar el tipo y tamaño del frame
    # stframe.write(f"Tipo del frame antes de conversión: {frame.dtype}")
    # stframe.write(f"Tamaño del frame antes de conversión: {frame.shape}")
    
    if frame.dtype != np.uint8:
        frame_float32 = frame.astype(np.float32)  # Convierte el frame a float32
        min_val = frame_float32.min()  # Encuentra el valor mínimo
        max_val = frame_float32.max()  # Encuentra el valor máximo
        # Evitar división por cero
        if max_val - min_val == 0:
            frame_normalizado = np.zeros_like(frame_float32)  # Si todos los valores son iguales, establece a cero
        else:
            frame_normalizado = (frame_float32 - min_val) / (max_val - min_val)  # Normaliza el frame
            
        frame_uint8 = (frame_normalizado * 255).astype(np.uint8)  # Escala a rango [0, 255]
    else:
        # Si ya está en uint8, no es necesario normalizar
        frame_uint8 = frame

    # Verificar si el frame es en escala de grises antes de convertir a RGB
    if len(frame_uint8.shape) == 2:  # Imágenes en escala de grises tienen 2 dimensiones
        frame_nd2 = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2BGR)  # Convierte a BGR
    else:
        frame_nd2 = frame_uint8  # Mantiene el frame original

    # st.write(f"Tipo del frame después de conversión: {frame_rgb.dtype}")
    # st.write(f"Tamaño del frame después de conversión: {frame_rgb.shape}")
    
    return frame_nd2  # Retorna el frame convertido

def guardar_y_eliminar_video_temporal(output_file):
    clip = VideoFileClip(output_file.name)  # Carga el video de salida
    clip.write_videofile(output_file.name, codec='libx264', audio_codec='aac')  # Guarda el video con el codec especificado
    
    with open(output_file.name, "rb") as f:
        video_data = f.read()  # Lee los datos del video
    
    # Opción para mostrar el video (si es necesario)
    # st.video(video_data, format="video/mp4")
    
    try:
        if output_file:
            output_file.close()  # Cierra el archivo antes de eliminarlo
            os.remove(output_file.name)  # Elimina archivo temporal
            # st.write("Archivo eliminado exitosamente.")
    except Exception as e:
        st.warning(f"No se pudo eliminar el archivo: {e}")  # Muestra una advertencia si no se pudo eliminar el archivo


def guardar_resultados(video_bytes, video_info, sperm_counts, track_history, last_frame, bbox_sizes, trajectory_data, 
                       video_byte_list, all_video_info, all_sperm_counts, all_track_histories, all_last_frames, all_bbox_sizes, all_trajectory_data):
    # Guardar los resultados en las listas correspondientes
    video_byte_list.append(video_bytes)
    all_video_info.append(video_info)
    all_sperm_counts.append(sperm_counts)
    all_track_histories.append(track_history)
    all_last_frames.append(last_frame)
    all_bbox_sizes.append(bbox_sizes)
    all_trajectory_data.append(trajectory_data)

    return video_byte_list, all_video_info, all_sperm_counts, all_track_histories, all_last_frames, all_bbox_sizes, all_trajectory_data