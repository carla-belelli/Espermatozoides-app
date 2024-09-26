La aplicacion permite analizar videos de espermatozoides para evaluar su movimiento y comportamiento. Utiliza YOLO como algoritmo de detección, OC-SORT o BoxMOT como algoritmo de seguimiento, para identificar y rastrear las trayectorias de los espermatozoides. Ofrece una interfaz interactiva en Streamlit donde se pueden cargar videos, ajustar parámetros de análisis, y visualizar resultados como trayectorias, parámetros cinéticos y categorías de movimiento. Es una herramienta útil para investigaciones científicas en el área de la biología y salud reproductiva.

1. Clonar el Repositorio
Abrir una terminal y clonar el repositorio:
    git clone https://github.com/carla-belelli/Espermatozoides-app.git

Navegar al directorio del proyecto:
    cd Espermatozoides-app

2. Crear y Activar un Entorno Virtual
Crea un entorno virtual:
    python -m venv env
Activa el entorno virtual:
    Windows:
        .\env\Scripts\activate
    macOS/Linux:
        source env/bin/activate

3. Instalar Dependencias

Instala las dependencias del proyecto:
    pip install -r requirements.txt

4. Ejecutar la Aplicación

Inicia la aplicación con Streamlit:
    streamlit run app.py

Recomendaciones
Asegúrate de tener Python 3.8 o superior instalado.
Si necesitas un entorno de desarrollo, puedes usar Visual Studio Code con la extensión de Python.
Para macOS, debes tener instalado ffmpeg, puedes instalarlo con brew: brew install ffmpeg