import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from collections import Counter
from pathlib import Path
import tempfile
import os
from boxmot import BoTSORT
import matplotlib.pyplot as plt
import pandas as pd
from moviepy.editor import VideoFileClip
from boxmot import HybridSORT
from boxmot import OCSORT
import io
from nd2reader import ND2Reader
from fpdf import FPDF
import zipfile
from io import BytesIO
from PIL import Image
import torch
from moviepy.editor import ImageSequenceClip

# Función para calcular la distancia lineal entre dos puntos
def calcular_distancia_lineal(p1, p2):
    """
    Calcula la distancia lineal entre dos puntos p1 y p2.
    
    Args:
        p1: Tupla o lista con coordenadas (x, y) del primer punto.
        p2: Tupla o lista con coordenadas (x, y) del segundo punto.
    
    Returns:
        Distancia lineal entre los puntos p1 y p2.
    """
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)  # Teorema de Pitágoras

# Función para calcular la distancia curvilínea de una trayectoria
def calcular_distancia_curvilinea(trayectoria):
    """
    Calcula la distancia curvilínea de una trayectoria.
    
    Args:
        trayectoria: Lista de puntos (x, y) que definen la trayectoria.
    
    Returns:
        Distancia curvilínea total de la trayectoria.
    """
    distancia = 0  # Inicializa la distancia
    for i in range(1, len(trayectoria)):
        distancia += calcular_distancia_lineal(trayectoria[i-1], trayectoria[i])  # Suma las distancias lineales entre puntos
    return distancia  # Retorna la distancia total

# Función para calcular la linealidad de una trayectoria
def calcular_linealidad(trayectoria):
    """
    Calcula la linealidad de una trayectoria.
    
    Args:
        trayectoria: Lista de puntos (x, y) que definen la trayectoria.
    
    Returns:
        Valor de linealidad (entre 0 y 1) de la trayectoria.
    """
    if len(trayectoria) < 2:
        return 0  # Si hay menos de 2 puntos, no se puede calcular la linealidad
    distancia_lineal = calcular_distancia_lineal(trayectoria[0], trayectoria[-1])  # Distancia lineal entre el primer y el último punto
    distancia_curvilinea = calcular_distancia_curvilinea(trayectoria)  # Distancia curvilínea total
    if distancia_curvilinea == 0:
        return np.nan  # Evita división por cero
    else:
        return distancia_lineal / distancia_curvilinea  # Retorna la relación entre distancia lineal y curvilínea

# Función para calcular la dimensión fractal de una trayectoria
def calcular_dimension_fractal(trayectoria):
    """
    Calcula la dimensión fractal de una trayectoria.
    
    Args:
        trayectoria: Lista de puntos (x, y) que definen la trayectoria.
    
    Returns:
        Dimensión fractal de la trayectoria.
    """
    linealidad = calcular_linealidad(trayectoria)  # Calcula la linealidad
    n = len(trayectoria)  # Número de puntos en la trayectoria
    if linealidad == 0:
        return np.nan  # Evita cálculo en caso de linealidad cero
    else:
        return np.log(n) / np.log(n * linealidad)  # Calcula la dimensión fractal

# Función para calcular la velocidad promedio lineal
def calcular_velocidad_lineal(distancia_lineal, fps, n_puntos, tiempo_total):
    """
    Calcula la velocidad promedio lineal.
    
    Args:
        distancia_lineal: Distancia lineal total recorrida.
        fps: Frames por segundo del video.
        n_puntos: Número de puntos en la trayectoria.
        tiempo_total: Tiempo total de observación.
    
    Returns:
        Velocidad promedio lineal.
    """
    tiempo_observación = n_puntos / fps  # Calcula el tiempo de observación basado en el número de puntos y fps
        
    if tiempo_total > 0:
        return distancia_lineal / tiempo_observación  # Retorna la velocidad promedio lineal
    else:
        return np.nan  # Evita cálculo en caso de tiempo total cero

# Función para calcular la velocidad promedio curvilínea
def calcular_velocidad_curvilinea(distancia_curvilinea, fps, n_puntos, tiempo_total):
    """
    Calcula la velocidad promedio curvilínea.
    
    Args:
        distancia_curvilinea: Distancia curvilínea total recorrida.
        fps: Frames por segundo del video.
        n_puntos: Número de puntos en la trayectoria.
        tiempo_total: Tiempo total de observación.
    
    Returns:
        Velocidad promedio curvilínea.
    """
    tiempo_observación = n_puntos / fps  # Calcula el tiempo de observación basado en el número de puntos y fps
    if tiempo_total > 0:
        return distancia_curvilinea / tiempo_observación  # Retorna la velocidad promedio curvilínea
    else:
        return np.nan  # Evita cálculo en caso de tiempo total cero
