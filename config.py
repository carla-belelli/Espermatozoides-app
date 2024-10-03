import streamlit as st
from pathlib import Path
from boxmot import BoTSORT
from boxmot import OCSORT
import torch

# Función para cargar los archivos de video
def cargar_videos():
    # Crear un cargador de archivos en la barra lateral para permitir la carga de videos .nd2
    uploaded_files = st.sidebar.file_uploader("Sube un video", type=["nd2"], accept_multiple_files=True)
    return uploaded_files  # Retornar los archivos subidos

# Función para verificar si usar GPU o CPU
def seleccionar_dispositivo():
    # Crear una casilla de verificación en la barra lateral para verificar la disponibilidad de GPU
    print(torch.cuda.is_available())
    check_gpu = st.sidebar.checkbox("¿Verificar disponibilidad de GPU?", value=False)

    if check_gpu:
        # Si el usuario desea verificar la GPU, comprobar si CUDA está disponible
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()  # Contar cuántas GPUs están disponibles
            if num_gpus > 0:
                device = torch.device("cuda:0")  # Usar la primera GPU disponible
                gpu_name = torch.cuda.get_device_name(0)  # Obtener el nombre de la GPU
                st.sidebar.write(f"Usando GPU: {gpu_name}")  # Mostrar el nombre de la GPU en la barra lateral
            else:
                st.sidebar.write("No hay GPUs disponibles, usando CPU.")  # Avisar que no hay GPUs
                device = torch.device("cpu")  # Usar CPU si no hay GPUs
        else:
            device = torch.device("cpu")  # Usar CPU si CUDA no está disponible
            st.sidebar.write("CUDA no está disponible, usando CPU.")
    else:
        device = torch.device("cpu")  # Usar CPU si el usuario no desea verificar la GPU

    return device  # Retornar el dispositivo seleccionado (GPU o CPU)

# Función para configurar parámetros generales (confidence, num_frames, etc.)
def configurar_parametros():
    # Crear un control deslizante para ajustar el nivel de confianza
    confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.1)
    # Permitir al usuario ingresar el número de frames a procesar
    num_frames = st.sidebar.number_input("Número de frames a procesar", min_value=1, value=100)
    # Permitir al usuario definir el número mínimo de puntos en la trayectoria
    min_puntos_trayectoria = st.sidebar.number_input("Número de puntos mínimos en la trayectoria", min_value=1, value=50)
    # Permitir al usuario establecer la conversión de píxeles a micrómetros
    pixeles_a_micrómetros = st.sidebar.number_input("Conversión de píxeles a micrómetros", min_value=1.0, value=1.30)
    # Permitir al usuario definir la distancia umbral para las trayectorias
    max_dist_threshold = st.sidebar.number_input("Distancia umbral para trayectoria", min_value=1, value=5)
    
    # Retornar todos los parámetros configurados
    return confidence, num_frames, min_puntos_trayectoria, pixeles_a_micrómetros, max_dist_threshold

# Función para seleccionar e inicializar el tracker
def seleccionar_tracker():
    
    tracker_option = st.sidebar.selectbox("Seleccione el tracker:", ["OCSORT", "BoTSORT"])

    return tracker_option

# Función separada para quitar el fondo estático
def pregunta_quitar_fondo(uploaded_files):
    # Crear una casilla de verificación en la barra lateral para preguntar si se desea quitar el fondo estático
    quitar_fondo = st.sidebar.checkbox("¿Desea quitar el fondo estático?", value=False)
    videos_seleccionados = []  # Lista para almacenar los videos seleccionados

    # Si el usuario desea quitar el fondo y se han subido archivos
    if quitar_fondo and uploaded_files:
        # Si hay más de un video subido, permitir la selección de videos
        if len(uploaded_files) > 1:
            st.sidebar.write("Selecciona los videos a los que deseas quitar el fondo estático:")

            # Iterar a través de los archivos subidos para crear casillas de verificación
            for file in uploaded_files:
                if st.sidebar.checkbox(f"{file.name}", value=False):
                    videos_seleccionados.append(file.name)  # Agregar el video seleccionado a la lista
    return quitar_fondo, videos_seleccionados  # Retornar la opción de quitar fondo y la lista de videos seleccionados





