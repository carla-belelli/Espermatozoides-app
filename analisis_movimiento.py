import streamlit as st
import cv2
import numpy as np
from collections import Counter
import os
import pandas as pd
from calculo_parametros import (
    calcular_distancia_lineal,
    calcular_distancia_curvilinea,
    calcular_linealidad,
    calcular_dimension_fractal,
    calcular_velocidad_lineal,
    calcular_velocidad_curvilinea
)

def procesar_detecciones_y_caracteristicas(video_info, uploaded_files, idx, output_file, sperm_counts, track_history, bbox_sizes, num_frames, min_puntos_trayectoria, calcular_distancia_lineal, calcular_dimension_fractal, pixeles_a_micrómetros, fps, tiempo_total):
    # 1. Mostrar características del video procesado
    file_name, _ = os.path.splitext(uploaded_files[idx].name)
    st.subheader(f"Características del Video: {file_name}")
    for key, value in video_info.items():
        st.markdown(f"- **{key}:** {value}")
    
    # 2. Calcular la moda y media de los conteos de espermatozoides
    st.write(sperm_counts)
    if sperm_counts:
        contador = Counter(sperm_counts)
        moda_valor = [num for num, freq in contador.items() if freq == contador.most_common(1)[0][1]]
        moda_unica = moda_valor[0]  # Tomar solo una de las modas, si hay más de una
        media = np.mean(sperm_counts)
        st.subheader('Resultados:')
        st.markdown(f"- **Se ha encontrado en promedio** {int(media)} **espermatozoides por frame**")
    
    # 3. Mostrar IDs con bbox repetidos
    present_ids = list(track_history.keys())
    ids_con_bbox_repetido = []
    for id_, sizes in bbox_sizes.items():
        # Contar las repeticiones de cada tamaño de bbox
        size_counts = Counter(sizes)
        
        # Ajustar el valor de x basado en num_frames
        if num_frames <= 150:
            x = num_frames / 5
        else:
            x = num_frames / 10  # Valor predeterminado para num_frames > 500
        
        # Verificar si algún tamaño de bbox se repite más de x veces
        if any(count > x for count in size_counts.values()):
            ids_con_bbox_repetido.append(id_)
    
    st.markdown(f"- **IDs Muertos** {len(ids_con_bbox_repetido)}: {ids_con_bbox_repetido}")
    
    # 4. Filtrar IDs inmóviles y progresivos
    cantidad_ids = len(track_history)
    
    present_ids_inmoviles = [id_ for id_ in present_ids if len(track_history[id_]) > 1 and
                             (calcular_distancia_lineal(track_history[id_][0], track_history[id_][-1]) * pixeles_a_micrómetros) < 8 and
                             id_ not in ids_con_bbox_repetido]
    
    st.markdown(f"- **De {cantidad_ids} IDs hay** {len(present_ids_inmoviles)} **que poseen cabeza fija. IDs:** {present_ids_inmoviles}")
    
    present_ids_con_suficientes_puntos = [id_ for id_ in present_ids if len(track_history[id_]) >= min_puntos_trayectoria and
                                          id_ not in ids_con_bbox_repetido and
                                          id_ not in present_ids_inmoviles]
    
    st.markdown(f"- **De {cantidad_ids} IDs solo** {len(present_ids_con_suficientes_puntos)} **presentan al menos {min_puntos_trayectoria} puntos y son móviles. IDs:** {present_ids_con_suficientes_puntos}")
    
    present_ids_mobiles_e_inmoviles = set(present_ids_con_suficientes_puntos) | set(present_ids_inmoviles) | set(ids_con_bbox_repetido)
    present_ids_sobrantes = set(present_ids) - present_ids_mobiles_e_inmoviles
    st.markdown(f"- **De {cantidad_ids} IDs, hay** {len(present_ids_sobrantes)} **que no cumplen con ninguna de las dos condiciones. IDs:** {list(present_ids_sobrantes)}")
    
    return moda_unica, media, cantidad_ids, present_ids_sobrantes, present_ids_con_suficientes_puntos, present_ids_inmoviles, ids_con_bbox_repetido

def calcular_y_clasificar_trayectorias(track_history, present_ids_con_suficientes_puntos, pixeles_a_micrómetros, fps, tiempo_total):
    categorias = {
        'Movimiento Lineal': [],
        'Movimiento Transicional': [],
        'Movimiento Hiperactivado': []
    }
    dimensiones_fractales = {}
    datos_trayectorias = []
    total_velocidad_lineal = 0
    total_velocidad_curvilinea = 0
    num_trayectorias = 0
    
    # Rangos para cada categoría de movimiento
    rango_lineal = (1, 1.3)
    rango_transicional = (1.3, 1.8)
    rango_hiperactivado = (1.8, float('inf'))

    # Calcular dimensiones fractales y clasificar trayectorias
    for id_ in present_ids_con_suficientes_puntos:
        trayectoria = track_history[id_]
        dimension_fractal = calcular_dimension_fractal(trayectoria)
        dimensiones_fractales[id_] = dimension_fractal

        # Clasificar según la dimensión fractal
        if rango_lineal[0] <= dimension_fractal <= rango_lineal[1]:
            categorias['Movimiento Lineal'].append(id_)
        elif rango_transicional[0] <= dimension_fractal <= rango_transicional[1]:
            categorias['Movimiento Transicional'].append(id_)
        elif dimension_fractal >= rango_hiperactivado[0]:
            categorias['Movimiento Hiperactivado'].append(id_)

        # Calcular distancias, linealidad, y velocidades
        if trayectoria:  
            distancia_lineal = calcular_distancia_lineal(trayectoria[0], trayectoria[-1])
            distancia_curvilinea = calcular_distancia_curvilinea(trayectoria)
            linealidad = calcular_linealidad(trayectoria)
            n_puntos = len(trayectoria)

            # Velocidades
            velocidad_lineal = calcular_velocidad_lineal(distancia_lineal, fps, n_puntos, tiempo_total)
            velocidad_curvilinea = calcular_velocidad_curvilinea(distancia_curvilinea, fps, n_puntos, tiempo_total)

            # Cambiar unidad a micrómetros
            distancia_linealmicrometros = distancia_lineal * pixeles_a_micrómetros
            distancia_curvilineamicrometros = distancia_curvilinea * pixeles_a_micrómetros
            velocidad_linealmicrometros = velocidad_lineal * pixeles_a_micrómetros
            velocidad_curvilineamicrometros = velocidad_curvilinea * pixeles_a_micrómetros

            # Acumular velocidades totales
            total_velocidad_lineal += velocidad_linealmicrometros
            total_velocidad_curvilinea += velocidad_curvilineamicrometros
            num_trayectorias += 1

            # Determinar tipo de movimiento
            if id_ in categorias['Movimiento Lineal']:
                tipo_movimiento = 'Movimiento Lineal'
            elif id_ in categorias['Movimiento Transicional']:
                tipo_movimiento = 'Movimiento Transicional'
            elif id_ in categorias['Movimiento Hiperactivado']:
                tipo_movimiento = 'Movimiento Hiperactivado'
            else:
                tipo_movimiento = 'No Determinado'

            # Guardar datos de la trayectoria
            datos_trayectorias.append({
                'ID  ': id_,
                'Coef. de Dimensión Fractal': dimension_fractal,
                'Dist. Lineal [µm]': distancia_linealmicrometros,
                'Dist. Curvilínea [µm]': distancia_curvilineamicrometros,
                'Linealidad': linealidad,
                'N° de Puntos': n_puntos,
                'Vel. Lineal [µm/s]': velocidad_linealmicrometros,
                'Vel. Curvilínea [µm/s]': velocidad_curvilineamicrometros,
                'Tipo de Movimiento': tipo_movimiento
            })

    return datos_trayectorias, categorias, total_velocidad_lineal, total_velocidad_curvilinea, num_trayectorias


def generar_estadisticas_trayectorias(total_velocidad_lineal, total_velocidad_curvilinea, num_trayectorias, present_ids_con_suficientes_puntos, categorias):
    # Calcular promedios de velocidad
    if num_trayectorias > 0:
        promedio_velocidad_lineal = total_velocidad_lineal / num_trayectorias
        promedio_velocidad_curvilinea = total_velocidad_curvilinea / num_trayectorias
    else:
        promedio_velocidad_lineal = 0
        promedio_velocidad_curvilinea = 0

    # Contar espermatozoides por categoría
    conteo_categorias = {categoria: len(ids) for categoria, ids in categorias.items()}

    # Calcular porcentajes
    total_ids = len(present_ids_con_suficientes_puntos)
    porcentaje_lineal = (conteo_categorias.get('Movimiento Lineal', 0) / total_ids) * 100 if total_ids > 0 else 0
    porcentaje_transicional = (conteo_categorias.get('Movimiento Transicional', 0) / total_ids) * 100 if total_ids > 0 else 0
    porcentaje_hiperactivado = (conteo_categorias.get('Movimiento Hiperactivado', 0) / total_ids) * 100 if total_ids > 0 else 0

    # Organizar datos para visualización
    df_porcentajes = pd.DataFrame({
        'Movimiento': ['Lineal', 'Transicional', 'Hiperactivado'],
        'Porcentaje %': [porcentaje_lineal, porcentaje_transicional, porcentaje_hiperactivado]
    })

    # Mostrar promedio de velocidades
    st.subheader('Promedio de Velocidades [µm/s]')
    st.markdown(f"- **Promedio de Velocidad Lineal [µm/s]:** {promedio_velocidad_lineal:.2f}")
    st.markdown(f"- **Promedio de Velocidad Curvilínea [µm/s]:** {promedio_velocidad_curvilinea:.2f}")

    # Mostrar el conteo de espermatozoides por cada categoría
    st.subheader('Resultados de la Trayectoria de Espermatozoides: ')
    st.write(f"De {len(present_ids_con_suficientes_puntos)} espermatozoides, se han encontrado {conteo_categorias.get('Movimiento Lineal', 0)} espermatozoides que presentan trayectoria lineal, "
    f"{conteo_categorias.get('Movimiento Transicional', 0)} transicional, y "
    f"{conteo_categorias.get('Movimiento Hiperactivado', 0)} hiperactivado.")
    
    #st.write("Porcentaje de cada tipo de movimiento:")
    st.markdown("<u>Porcentaje de cada tipo de movimiento:</u>", unsafe_allow_html=True)
    st.dataframe(df_porcentajes)

    return promedio_velocidad_lineal, promedio_velocidad_curvilinea, conteo_categorias, df_porcentajes

def dibujar_trayectorias(last_frame, categorias, present_ids_con_suficientes_puntos, track_history):
    """
    Dibuja las trayectorias de cada tipo de movimiento en el último frame.

    Parámetros:
    - last_frame: El último frame sobre el cual se dibujarán las trayectorias.
    - categorias: Un diccionario que mapea los tipos de movimiento a los IDs de las trayectorias.
    - present_ids_con_suficientes_puntos: Una lista de IDs que tienen suficientes puntos para ser considerados.
    - track_history: Un diccionario que contiene el historial de trayectorias para cada ID.
    """
    # Dibujar las trayectorias de cada tipo de movimiento en diferentes colores
    for movimiento, ids in categorias.items():
        # Asignar color según el tipo de movimiento
        if movimiento == 'Movimiento Transicional':
            color = (255, 255, 0)  # Amarillo
        elif movimiento == 'Movimiento Hiperactivado':
            color = (0, 0, 255)  # Azul
        else:
            color = (178, 2, 86)  # Rojo por defecto

        # Asegurarse de que solo se dibujen trayectorias que cumplen con las condiciones
        for id_ in ids:
            if id_ in present_ids_con_suficientes_puntos and id_ in track_history:
                trayectoria = track_history[id_]
                trayectoria_array = np.array(trayectoria, dtype=np.int32)
                cv2.polylines(last_frame, [trayectoria_array], isClosed=False, color=color, thickness=1)