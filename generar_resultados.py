import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF
import zipfile
from PIL import Image
import io
from moviepy.editor import ImageSequenceClip


def generar_zipfile(report_paths, video_bytes_list, excel_buffers, uploaded_files, videoscolores):
    zip_path = 'Análisis_completo_videos.zip'  # Un archivo ZIP que contendrá todos los videos e informes
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for idx, report_path in enumerate(report_paths):
            file_name, _ = os.path.splitext(uploaded_files[idx].name)  # Obtener el nombre del archivo original (sin la extensión)
            
            # Crear una carpeta para el video
            folder_name = file_name
            zipf.writestr(f'{folder_name}/', '')  # Crear la carpeta vacía en el ZIP
            
            # Añadir informe PDF al ZIP
            informe_name = f'{folder_name}/{file_name}_informe.pdf'
            zipf.write(report_path, informe_name)
            
            # Añadir Excel al ZIP
            excel_buffer = excel_buffers[idx]
            excel_filename = f'{folder_name}/{file_name}_Detecciónyseguimiento.xlsx'
            zipf.writestr(excel_filename, excel_buffer.getvalue())
            
            # Añadir video original al ZIP
            video_bytes = video_bytes_list[idx]
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video1:
                temp_video1.write(video_bytes)
                temp_video1.flush()  # Asegúrate de escribir todo en disco
                temp_video1_path = temp_video1.name
                video_name_in_zip = f'{folder_name}/{file_name}.mp4'
                zipf.write(temp_video1_path, video_name_in_zip)
            
            # Añadir video con trayectorias coloreadas al ZIP
            video_color_path = videoscolores[idx]
            if os.path.exists(video_color_path):
                color_video_name_in_zip = f'{folder_name}/{file_name}_Trayectorias_colores.mp4'
                zipf.write(video_color_path, color_video_name_in_zip)
            else:
                print(f"Advertencia: El archivo {video_color_path} no existe y no se añadirá al ZIP.")

            # Eliminar archivos temporales creados
            os.remove(temp_video1_path)
            # Asegúrate de que video_color_path no sea None y exista
            if os.path.exists(video_color_path):
                os.remove(video_color_path)

    return zip_path

def crear_video_con_trayectorias(frames_a_colorear, categorias, present_ids_con_suficientes_puntos, track_history, output_pathvideo, fps, num_frames):
    # Crear un mapa de colores para los tipos de movimiento
    color_map = {
        'Movimiento Lineal': (86, 2, 178),      # Rojo
        'Movimiento Transicional': (0, 255, 255), # Amarillo
        'Movimiento Hiperactivado': (255, 0, 0)  # Azul
    }
    
    # Inicializar un diccionario para las trayectorias acumulativas
    trayectorias_acumulativas = {id_: [] for id_ in present_ids_con_suficientes_puntos}
    
    # Lista para almacenar los frames procesados
    frames_procesados = []
    
    # Procesar cada frame
    for frame_number in range(min(num_frames, len(frames_a_colorear))):
        frame = frames_a_colorear[frame_number].copy()
        
        # Actualizar las trayectorias acumulativas
        for id_ in trayectorias_acumulativas.keys():
            if id_ in track_history:
                trayectoria = track_history[id_]
                # Agregar el punto del frame actual a la trayectoria acumulativa
                if frame_number < len(trayectoria):
                    trayectorias_acumulativas[id_].append(trayectoria[frame_number])
        
        # Dibujar las trayectorias en el frame
        for movimiento, ids in categorias.items():
            color = color_map.get(movimiento, (0, 255, 0))  # Verde por defecto si no está en el mapa
            for id_ in ids:
                if id_ in trayectorias_acumulativas and len(trayectorias_acumulativas[id_]) > 1:
                    trayectoria_array = np.array(trayectorias_acumulativas[id_], dtype=np.int32)
                    cv2.polylines(frame, [trayectoria_array], isClosed=False, color=color, thickness=1)
        
        # Convertir el frame de BGR a RGB para MoviePy
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Agregar el frame procesado a la lista
        frames_procesados.append(frame_rgb)
    
    # Crear un video a partir de la secuencia de imágenes procesadas
    clip = ImageSequenceClip(frames_procesados, fps=fps)
    
    # Guardar el video usando MoviePy
    clip.write_videofile(output_pathvideo, codec="libx264", audio=False)
    
    return output_pathvideo

def dividir_lista_en_lineas(lista, max_caracteres_por_linea):
    lineas = []
    linea_actual = ""

    for item in lista:
        item_str = str(item)
        if len(linea_actual) + len(item_str) + 2 > max_caracteres_por_linea:
            lineas.append(linea_actual)
            linea_actual = item_str
        else:
            if linea_actual:
                linea_actual += ", " + item_str
            else:
                linea_actual = item_str

    if linea_actual:
        lineas.append(linea_actual)

    return lineas

def generar_informe(video_info, idx,
                        sperm_counts,
                        track_history,
                        bbox_sizes,
                        moda_unica,
                        media,
                        cantidad_ids,
                        ids_con_bbox_repetido,
                        min_puntos_trayectoria,
                        present_ids_inmoviles,
                        present_ids_con_suficientes_puntos,
                        present_ids_sobrantes,
                        df_trayectorias,
                        df_porcentajes,
                        fig,
                        promedio_velocidad_lineal,
                        promedio_velocidad_curvilinea,
                        conteo_categorias,last_frame, output_dir="informes"):
    
    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Inicializar el PDF
    pdf = FPDF('L', 'mm', 'A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Título del informe
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, 'Informe de Análisis de Videos de Espermatozoides', 0, 1, 'C')

    # Información general del video
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, f'Características del Video:', 0, 1)

    pdf.set_font("Arial", '', 12)
    for key, value in video_info.items():
                pdf.cell(0, 10, f"{key}: {value}", 0, 1)

    pdf.ln(10)  # Añadir un espacio vertical

    # Análisis de espermatozoides
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, 'Análisis de Espermatozoides:', 0, 1)

    
    if sperm_counts:
            pdf.set_font("Arial", '', 12)
            #pdf.cell(0, 10, f"La moda de detecciones por frame es: {moda_unica}", 0, 1)
            pdf.cell(0, 10, f"Se ha encontrado en promedio {int(media)} espermatozoides por frame", 0, 1)
            pdf.ln(5)

    # IDs de Espermatozoides clasificados y análisis de trayectorias
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, 'Análisis de Trayectorias:', 0, 1)
    
    pdf.set_font("Arial", '', 12)
    cantidad_ids = len(track_history)
    pdf.cell(0, 10, f"Cantidad total de IDs obtenidos: {cantidad_ids}", 0, 1)
    
    pdf.cell(0, 10, f"De {cantidad_ids} IDs hay {len(ids_con_bbox_repetido)} muertos. IDs:", 0, 1)
    #pdf.cell(0, 10, f"De {cantidad_ids} IDs hay {ids_con_bbox_repetido} muertos. IDs:", 0, 1) ##ESTO NO VAA
    for linea in dividir_lista_en_lineas(ids_con_bbox_repetido, max_caracteres_por_linea=150):
                pdf.cell(0, 10, linea, 0, 1)

            # IDs con movimiento inmóvil
    pdf.cell(0, 10, f"De {cantidad_ids} IDs hay {len(present_ids_inmoviles)} que poseen cabeza fija. IDs:", 0, 1)
    for linea in dividir_lista_en_lineas(present_ids_inmoviles, max_caracteres_por_linea=150):
                pdf.cell(0, 10, linea, 0, 1)

            # IDs móviles progresivos
    pdf.cell(0, 10, f"De {cantidad_ids} IDs hay {len(present_ids_con_suficientes_puntos)} que presentan al menos {min_puntos_trayectoria} puntos y son moviles progresivos para clasificar. IDs:", 0, 1)
    for linea in dividir_lista_en_lineas(present_ids_con_suficientes_puntos, max_caracteres_por_linea=150):
                pdf.cell(0, 10, linea, 0, 1)

            # IDs no móviles ni inmóviles
    pdf.cell(0, 10, f"De {cantidad_ids} IDs, hay {len(present_ids_sobrantes)} que no cumplen con ninguna de las dos condiciones (no móviles ni inmóviles). IDs:", 0, 1)
    for linea in dividir_lista_en_lineas(present_ids_sobrantes, max_caracteres_por_linea=150):
                pdf.cell(0, 10, linea, 0, 1)
            
    pdf.ln(5)
        # Agregar tabla de trayectorias en el PDF
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, 'Datos de Trayectorias:', 0, 1)
    
    # Definir columnas y calcular el ancho
    columns = df_trayectorias.columns.tolist()
    col_widths = [max(len(str(col)), max(df_trayectorias[col].astype(str).apply(len))) * 1.8 for col in columns]  # Ajuste automático del ancho de columna
    
    # Encabezados de columna
    pdf.set_font("Arial", 'B', 9)
    for i, column in enumerate(columns):
        pdf.cell(col_widths[i], 10, column, 1, 0, 'L')
    pdf.ln()

    # Datos de la tabla
    pdf.set_font("Arial", '', 10)
    for row in df_trayectorias.itertuples(index=False):
        for i, value in enumerate(row):
            pdf.cell(col_widths[i], 10, str(value), 1)
        pdf.ln()
    
    pdf.ln(10)  # Añadir un espacio vertical

    # Agregar promedios de velocidad
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, 'Promedio de Velocidades:', 0, 1)
    
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Promedio de Velocidad Lineal [µm/s]: {promedio_velocidad_lineal:.2f}", 0, 1)
    pdf.cell(0, 10, f"Promedio de Velocidad Curvilínea [µm/s]: {promedio_velocidad_curvilinea:.2f}", 0, 1)
    pdf.ln(10)  # Añadir un espacio vertical

    # Agregar cantidad de espermatozoides por categoría
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, 'Cantidad de Espermatozoides por Categoría:', 0, 1)
    
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"De {len(present_ids_con_suficientes_puntos)} espermatozoides con suficientes puntos, se han encontrado {conteo_categorias.get('Movimiento Lineal', 0)} que presentan trayectoria lineal, "
                    f"{conteo_categorias.get('Movimiento Transicional', 0)} transicional, y "
                    f"{conteo_categorias.get('Movimiento Hiperactivado', 0)} hiperactivado.", 0, 1)

    # Agregar porcentajes de movimiento
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, 'Porcentaje de Cada Tipo de Movimiento:', 0, 1)
    
    for index, row in df_porcentajes.iterrows():
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, f"{row['Movimiento']}: {row['Porcentaje %']:.2f}%", 0, 1)

    pdf.ln(10)  # Añadir un espacio vertical
    pdf.add_page()

    # Agregar gráfico de barras
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, 'Distribución de Categorías de Movimiento:', 0, 1)
    
    fig.savefig(os.path.join(output_dir, 'grafico_barras.png'))
    pdf.image(os.path.join(output_dir, 'grafico_barras.png'), x=10, y=pdf.get_y(), w=180)

    # Guardar último frame
    last_frame_path = os.path.join(output_dir, 'last_frame.png')
    last_frame_image = Image.fromarray(last_frame)
    last_frame_image.save(last_frame_path)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, 'Último Frame del Video:', 0, 1)
    pdf.image(last_frame_path, x=10, y=pdf.get_y(), w=180)

    # Guardar el informe
    report_path = os.path.join(output_dir, f"informe_analisisespermatozoides_{idx + 1}.pdf")
    pdf.output(report_path)

    # Limpiar archivos temporales
    #os.remove('grafico_barras.png')
    os.remove(last_frame_path)

    return report_path

def generar_dataframes_trayectorias(datos_trayectorias, trajectory_data, pixeles_a_micrómetros):
    # Convertir la lista de datos a un DataFrame de pandas
    df_trayectorias = pd.DataFrame(datos_trayectorias)
    trajectory_df = pd.DataFrame(trajectory_data)
    
    # Redondear los valores a 4 decimales
    df_trayectorias = df_trayectorias.round(4)
    df_trayectorias = df_trayectorias.sort_values(by=['ID  '])
    trajectory_df = trajectory_df.round(4)
    trajectory_df = trajectory_df.sort_values(by=['ID', 'FRAME'])
    
    # Crear un buffer para almacenar el archivo Excel
    buffer = io.BytesIO()
    
    # Escribir los DataFrames en el buffer como un archivo Excel
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Escribir el DataFrame de trayectorias en la hoja de cálculo 'TipodeMovimiento'
        df_trayectorias.to_excel(writer, sheet_name="TipodeMovimiento", index=False)
        # Escribir el DataFrame de coordenadas en la hoja de cálculo 'DatosCoordenadas'
        trajectory_df.to_excel(writer, sheet_name="DatosCoordenadas", index=False)
    
    # Mover el puntero al principio del buffer
    buffer.seek(0)

    # Mostrar el DataFrame en Streamlit
    st.subheader('Resumen de Parámetros y Tipos de Movimiento:')
    st.dataframe(df_trayectorias)
    st.subheader('Datos de Coordenadas:')
    st.dataframe(trajectory_df)
    
    # Devolver los DataFrames y el buffer
    return df_trayectorias, trajectory_df, buffer

##VER SI QUEREMOS O NO LA GRAFICA DE DISTRIBUCION CATEGORIAS
def graficar_distribucion_categorias(conteo_categorias):
    fig, ax = plt.subplots()
    ax.bar(conteo_categorias.keys(), conteo_categorias.values(), color=['red', 'yellow', 'blue'])
    ax.set_xticks(range(len(conteo_categorias)))
    ax.set_xticklabels(conteo_categorias.keys(), rotation=5, ha='right')
    ax.set_title('Distribución de Categorías de Movimiento')
    ax.set_xlabel('Categoría')
    ax.set_ylabel('Número de Espermatozoides')
    st.pyplot(fig)
    return fig