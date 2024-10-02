
# Espermatozoides-app

## Descripción

La aplicación permite analizar videos de espermatozoides para evaluar su movimiento y comportamiento. Utiliza YOLO como algoritmo de detección y OC-SORT o BoT-SORT como algoritmos de seguimiento para identificar y rastrear las trayectorias. Ofrece una interfaz interactiva en Streamlit para cargar videos, ajustar parámetros de análisis y visualizar resultados como trayectorias, parámetros cinéticos y categorías de movimiento. Es una herramienta útil para investigaciones en biología y salud reproductiva.

## Instrucciones de Instalación

### 1. Clonar el Repositorio
Abrir una terminal y clonar el repositorio:
```bash
git clone https://github.com/carla-belelli/Espermatozoides-app.git
```
Navegar al directorio del proyecto:
```bash
cd Espermatozoides-app
```

### 2. Crear y Activar un Entorno Virtual
Crea un entorno virtual:
```bash
py -m venv env
```
Activa el entorno virtual:
- **Windows**:
  ```bash
  .\env\Scripts\activate
  ```
- **macOS/Linux**:
  ```bash
  source env/bin/activate
  ```

### 3. Instalar Dependencias
Instala las dependencias del proyecto:
```bash
pip install -r requirements.txt
```

### 4. Ejecutar la Aplicación
Inicia la aplicación con Streamlit: 
```bash
streamlit run app.py
```

### Recomendaciones
- Asegúrate de tener Python 3.9 o superior y git. Puedes descargarlos mediante los siguients links: https://www.python.org/ftp/python/3.9.0/python-3.9.0-amd64.exe y https://git-scm.com/downloads 
- En macOS, instala `ffmpeg` con Homebrew:
  ```bash
  brew install ffmpeg
  ```

