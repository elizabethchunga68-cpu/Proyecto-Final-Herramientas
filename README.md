# Sistema de Detección de Seguridad Vial

Aplicación de visión artificial que utiliza un modelo YOLO para detectar en tiempo real:

- Casco de seguridad.
- Chaleco reflectivo.
- Señal de peligro.

El proyecto ofrece una interfaz web para visualizar el video, comprobar los elementos detectados y generar reportes de seguridad.

## Tecnologías utilizadas

- Python
- YOLO mediante Ultralytics
- OpenCV
- HTML, CSS y JavaScript
- DroidCam o cámara local

## Requisitos

- Windows 10 u 11
- Python 3.10 o superior
- Una cámara local o DroidCam
- Modelo YOLO `best.pt`

## Instalación

### 1. Clonar el repositorio

```powershell
git clone https://github.com/elizabethchunga68-cpu/Proyecto-Final-Herramientas.git
cd "Proyecto-Final-Herramientas"
```

### 2. Crear y activar el entorno virtual

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Instalar las dependencias

```powershell
pip install ultralytics opencv-python
```

Verifica que estén presentes los archivos `best.pt`, `detector.py`, `web_app.py` e `index.html`.

## Configuración de la cámara

La dirección predeterminada de DroidCam se encuentra en `detector.py` y `web_app.py`:

```python
DROIDCAM_SOURCE = "http://192.168.100.199:4747/video"
```

Reemplaza la dirección por la mostrada por la aplicación DroidCam.

Si DroidCam no está disponible, el sistema intentará usar una cámara local o virtual mediante los índices `0`, `1` y `2`.

## Uso desde la interfaz web

Ejecuta el servidor:

```powershell
python web_app.py
```

Luego abre en el navegador:

```text
http://127.0.0.1:8000
```

La interfaz permite:

- Ver el video procesado.
- Consultar los elementos detectados.
- Identificar los elementos faltantes.
- Generar un reporte manual.

## Uso desde la consola

También puedes iniciar el detector sin interfaz web:

```powershell
python detector.py
```

Controles:

- `q`: cerrar el programa.
- `r`: generar un reporte manual si faltan elementos.

## Reportes

El sistema genera automáticamente un reporte cuando la misma condición de elementos faltantes permanece durante al menos 5 segundos.

Los reportes se guardan en:

```text
reporte_seguridad_vial.txt
```

Cada reporte contiene:

- Fecha y hora.
- Fuente de video utilizada.
- Tiempo que se mantuvo el estado.
- Elementos detectados.
- Elementos no detectados.
- Recomendaciones de seguridad.

## Archivos principales

| Archivo | Descripción |
|---|---|
| `detector.py` | Detector ejecutado desde la consola |
| `web_app.py` | Servidor web y procesamiento con YOLO |
| `index.html` | Interfaz del panel de seguridad |
| `best.pt` | Modelo YOLO entrenado |
| `reporte_seguridad_vial.txt` | Historial de reportes |

## Funcionamiento

El programa procesa cada fotograma, clasifica las detecciones y conserva únicamente aquellas que superan los umbrales de confianza configurados. Para reducir falsos positivos, los elementos deben aparecer durante una cantidad consecutiva de fotogramas antes de considerarse detectados.

Los elementos faltantes se muestran en el panel y pueden generar automáticamente un reporte después de permanecer sin cambios durante el tiempo configurado.

## Configuración avanzada

Las siguientes constantes se pueden ajustar al inicio de `detector.py`:

- `CONF_THRESHOLD`: confianza general de detección.
- `MIN_DETECTIONS_CONSECUTIVAS_POR_CLASE`: fotogramas consecutivos requeridos por elemento.
- `MIN_CONFIDENCE_POR_CLASE`: confianza mínima por clase.
- `SEGUNDOS_SIN_DETECCION`: tiempo antes de crear el reporte automático.
