import streamlit as st
from io import BytesIO
from utils import mostrar_instrucciones, guardar_y_eliminar_video_temporal
from config import cargar_videos , seleccionar_dispositivo, configurar_parametros, seleccionar_tracker, pregunta_quitar_fondo
from deteccion_y_seguimiento import procesar_video
from analisis_movimiento import procesar_detecciones_y_caracteristicas , calcular_y_clasificar_trayectorias, generar_estadisticas_trayectorias, dibujar_trayectorias
from calculo_parametros import (
    calcular_distancia_lineal,
    calcular_dimension_fractal
)
from generar_resultados import (generar_dataframes_trayectorias, generar_informe , generar_zipfile, crear_video_con_trayectorias, graficar_distribucion_categorias)
from moviepy.editor import VideoFileClip

def main_app():
    st.title("Detección y Seguimiento de Espermatozoides")
    
    # Lista para almacenar las rutas de los archivos temporales
    video_paths = []
    uploaded_files = cargar_videos()
    device = seleccionar_dispositivo()
    confidence, num_frames, min_puntos_trayectoria, pixeles_a_micrómetros, max_dist_threshold = configurar_parametros()
    tracker = seleccionar_tracker(device)
    quitar_fondo, videos_seleccionados = pregunta_quitar_fondo(uploaded_files)
    
    # Crear espacio vacío para los frames
    stframe = st.empty()

    # Crear espacio para mostrar el número de frame que se está procesando
    progress_text = st.empty()
    # Crear archivo temporal para almacenar el video subido
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_type = uploaded_file.name.split('.')[-1]
            # Leer el contenido del archivo subido en memoria
            video = BytesIO(uploaded_file.read())
            video_paths.append(video)
    else:
        video_paths = []

    # Botón para iniciar inferencia
    if st.sidebar.button("Iniciar"):
        if video_paths:
                    # Listas para almacenar resultados de todos los videos
            video_byte_list = [] 
            all_video_info = []
            all_sperm_counts = []
            all_track_histories = []
            all_last_frames = []
            all_bbox_sizes = []
            report_paths = []
            excel_buffers = []
            all_trajectory_data = []
            videoscolores = []
                                
            #for idx, video_path in enumerate(video_paths):
            for idx, video_file in enumerate(uploaded_files):
                video_name = video_file.name  # Usar el nombre del archivo directamente

                # Determinar si este video debe quitarse el fondo comparando los nombres de los archivos
                quitar_fondo_video = video_name in videos_seleccionados  # Comparar por nombre de archivo
                
                #Procesar el video
                progress_text.text(f"Procesando video {idx + 1} de {len(uploaded_files)}...")
                video_bytes, sperm_counts, track_history, last_frame, output_file, tiempo_total, video_info, bbox_sizes, fps, trajectory_data, frames_a_colorear = procesar_video(video_file, confidence, stframe, progress_text, max_dist_threshold, num_frames=num_frames, device=device, quitar_fondo=quitar_fondo_video, tracker=tracker)
                # Agregar resultados a las listas
                video_byte_list.append(video_bytes)
                all_video_info.append(video_info)
                all_sperm_counts.append(sperm_counts)
                all_track_histories.append(track_history)
                all_last_frames.append(last_frame)
                all_bbox_sizes.append(bbox_sizes)
                all_trajectory_data.append(trajectory_data)

                guardar_y_eliminar_video_temporal(output_file)
                    # Limpiar el espacio de los frames individuales
                stframe.empty()
                moda_unica, media, cantidad_ids, present_ids_sobrantes, present_ids_con_suficientes_puntos, present_ids_inmoviles, ids_con_bbox_repetido = procesar_detecciones_y_caracteristicas(video_info, uploaded_files, idx, output_file, sperm_counts, track_history, bbox_sizes, num_frames, min_puntos_trayectoria, calcular_distancia_lineal, calcular_dimension_fractal, pixeles_a_micrómetros, fps, tiempo_total)
                        
                datos_trayectorias, categorias, total_velocidad_lineal, total_velocidad_curvilinea, num_trayectorias = calcular_y_clasificar_trayectorias(track_history, present_ids_con_suficientes_puntos, pixeles_a_micrómetros, fps, tiempo_total)

                promedio_velocidad_lineal, promedio_velocidad_curvilinea, conteo_categorias, df_porcentajes = generar_estadisticas_trayectorias(total_velocidad_lineal, total_velocidad_curvilinea, num_trayectorias, present_ids_con_suficientes_puntos, categorias)

                
                dibujar_trayectorias(last_frame, categorias, present_ids_con_suficientes_puntos, track_history)        
                                  
                # Generar los DataFrames y el buffer de Excel
                df_trayectorias, trajectory_df, buffer = generar_dataframes_trayectorias(datos_trayectorias, trajectory_data, pixeles_a_micrómetros)
                excel_buffers.append(buffer)  # Almacenar el buffer en la lista

                # Mostrar el cuadro procesado con las trayectorias categorizadas
                st.image(last_frame)

                fig = graficar_distribucion_categorias(conteo_categorias)

                # Guardar el video con trayectorias coloreadas
                output_pathvideo = f"Trayectorias_colores_video_{idx + 1}.mp4"
                videocolores = crear_video_con_trayectorias(frames_a_colorear, categorias, present_ids_con_suficientes_puntos, track_history, output_pathvideo, fps, num_frames)
                videoscolores.append(videocolores)
                # Leer el contenido del video
                with open(videocolores, "rb") as video_datax:
                    video_data = video_datax.read()

                # Mostrar el video en Streamlit
                st.video(video_data)
                
                # Después de procesar los videos y obtener toda la información
                output_pdf_path = f"informe_procesamiento_videos_{idx + 1}.pdf"
                report_path = generar_informe(video_info, idx, sperm_counts, track_history, bbox_sizes, moda_unica, media, cantidad_ids, ids_con_bbox_repetido, min_puntos_trayectoria, present_ids_inmoviles, present_ids_con_suficientes_puntos, present_ids_sobrantes, df_trayectorias, df_porcentajes, fig, promedio_velocidad_lineal, promedio_velocidad_curvilinea, conteo_categorias, last_frame, output_dir='informes')
                # Agregar el camino del informe a la lista
                report_paths.append(report_path)
                progress_text.empty()
            
            zip_path = generar_zipfile(report_paths,video_byte_list, excel_buffers,uploaded_files, videoscolores)
            with open(zip_path, "rb") as file:
                    st.download_button(
                        label="Descargar Informe y Video",
                        data=file,
                        file_name=zip_path,
                        mime="application/zip"
                    )
                
            st.write("Procesamiento completado.")
        #if not video_demo_shown:
            st.empty()  # Elimina el video de demostración de la pantalla principal

# Control de flujo de la aplicación
if 'instrucciones_vistas' not in st.session_state:
    mostrar_instrucciones()
    if st.button("Aceptar"):
        st.session_state.instrucciones_vistas = True
        st.rerun() 

# Mostrar la aplicación principal si las instrucciones ya han sido vistas
if st.session_state.get('instrucciones_vistas'):
    main_app()