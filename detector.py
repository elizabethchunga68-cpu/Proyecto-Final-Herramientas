import time
from datetime import datetime
from pathlib import Path
import cv2

# ==========================================
# CONFIGURACIÓN
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "best.pt")
CONF_THRESHOLD = 0.25
MIN_DETECTIONS_CONSECUTIVAS_POR_CLASE = {
    "casco": 8,
    "chaleco": 8,
    "senal": 20,
}
MIN_CONFIDENCE_POR_CLASE = {
    "casco": 0.60,
    "chaleco": 0.25,
    "senal": 0.90,
}

# URL de DroidCam desde tu iPhone o índice (1 / 2) de cámara virtual
DROIDCAM_SOURCE = "http://192.168.100.199:4747/video"

SEGUNDOS_SIN_DETECCION = 5.0
REPORTE_TXT = str(BASE_DIR / "reporte_seguridad_vial.txt")

# Ajusta estas palabras clave según el nombre exacto de tus clases en data.yaml
RECOMENDACIONES_POR_ELEMENTO = {
    "casco": "Casco: Verificar el uso obligatorio del casco de seguridad y su correcta colocación.",
    "chaleco": "Chaleco: Asegurar que el personal porte chaleco reflectivo visible.",
    "senal": "Señal de peligro: Colocar o verificar la visibilidad de la señal de peligro en la zona.",
}


def normalizar_nombre(nombre_clase: str) -> str:
    # Limpia tildes y minúsculas para emparejar fácilmente (ej. "señal_peligro" -> "senal_peligro")
    reemplazos = str.maketrans("áéíóúñ", "aeioun")
    return nombre_clase.lower().translate(reemplazos).strip().replace("-", "_").replace(" ", "_")


def clasificar_elemento(nombre_clase: str) -> str | None:
    nombre = normalizar_nombre(nombre_clase)

    if "casco" in nombre or "hard_hat" in nombre or nombre == "helmet":
        return "casco"
    if "chaleco" in nombre or nombre in {"vest", "safety_vest", "reflective_vest"}:
        return "chaleco"
    if "senal" in nombre or "peligro" in nombre or "warning_sign" in nombre:
        return "senal"

    return None


def identificar_faltantes(detectados_set: set[str]) -> list[str]:
    faltantes = []

    tiene_casco = any("casco" in d for d in detectados_set)
    tiene_chaleco = any("chaleco" in d for d in detectados_set)
    tiene_senal = any("senal" in d or "peligro" in d for d in detectados_set)

    if not tiene_casco:
        faltantes.append("casco")
    if not tiene_chaleco:
        faltantes.append("chaleco")
    if not tiene_senal:
        faltantes.append("senal")

    return faltantes


def generar_reporte(fuente: str, duracion: float, faltantes: list[str], presentes: set[str]):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    recomendaciones_lineas = "\n".join(f"- {RECOMENDACIONES_POR_ELEMENTO[item]}" for item in faltantes)
    presentes_str = ", ".join(sorted(presentes)) if presentes else "Ninguno"

    reporte = f"""
============================================================
REPORTE DE SEGURIDAD VIAL - ELEMENTOS NO DETECTADOS
============================================================
Fecha/Hora         : {timestamp}
Fuente (DroidCam)  : {fuente}
Tiempo sin cambio  : {duracion:.1f} s
Detectados         : {presentes_str}
No detectados      : {", ".join(faltantes)}

RECOMENDACIONES:
{recomendaciones_lineas}
============================================================
"""
    print(reporte)
    with open(REPORTE_TXT, "a", encoding="utf-8") as f:
        f.write(reporte + "\n")
    print(f"[INFO] Reporte guardado en '{REPORTE_TXT}'.")


def main():
    print("[INFO] Cargando modelo YOLO...", flush=True)
    from ultralytics import YOLO

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(DROIDCAM_SOURCE)

    if not cap.isOpened():
        print(f"[AVISO] No se pudo conectar a DroidCam en: {DROIDCAM_SOURCE}")
        cap.release()

        for indice in (0, 1, 2):
            cap = cv2.VideoCapture(indice)
            if cap.isOpened():
                print(f"[INFO] Usando cámara local o virtual con indice {indice}.")
                fuente_activa = f"camara_{indice}"
                break
        else:
            print("[ERROR] No se encontró ninguna cámara disponible.")
            print("[ERROR] Verifica que DroidCam esté iniciado y usa la misma red Wi-Fi.")
            return
    else:
        fuente_activa = DROIDCAM_SOURCE

    print("[INFO] Cámara conectada. Presiona 'q' para salir o 'r' para reportar faltantes.")

    ultimo_estado_faltantes = None
    inicio_estado = time.time()
    reporte_enviado_para_estado = False
    apariciones_consecutivas = {"casco": 0, "chaleco": 0, "senal": 0}

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.3)
            continue

        results = model.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            iou=0.55,
            max_det=20,
            verbose=False,
        )
        boxes = results[0].boxes

        detectados_raw = set()
        indices_validos = []
        if boxes is not None and len(boxes) > 0:
            for indice, box in enumerate(boxes):
                clase_nombre = model.names[int(box.cls[0])]
                confianza = float(box.conf[0])
                elemento = clasificar_elemento(clase_nombre)

                if elemento is not None and confianza >= MIN_CONFIDENCE_POR_CLASE[elemento]:
                    detectados_raw.add(elemento)
                    indices_validos.append(indice)

            results[0].boxes = boxes[indices_validos]

        detectados_estables = set()
        for elemento in ("casco", "chaleco", "senal"):
            if elemento in detectados_raw:
                apariciones_consecutivas[elemento] += 1
            else:
                apariciones_consecutivas[elemento] = 0

            if apariciones_consecutivas[elemento] >= MIN_DETECTIONS_CONSECUTIVAS_POR_CLASE[elemento]:
                detectados_estables.add(elemento)

        faltantes = identificar_faltantes(detectados_estables)
        ahora = time.time()

        # Si cambió qué elementos faltan, reiniciamos el temporizador
        if faltantes != ultimo_estado_faltantes:
            ultimo_estado_faltantes = faltantes
            inicio_estado = ahora
            reporte_enviado_para_estado = False

        # Solo reporta si faltan uno o más elementos
        if len(faltantes) > 0:
            tiempo_en_estado = ahora - inicio_estado
            if tiempo_en_estado >= SEGUNDOS_SIN_DETECCION and not reporte_enviado_para_estado:
                generar_reporte(fuente_activa, tiempo_en_estado, faltantes, detectados_estables)
                reporte_enviado_para_estado = True

            restante = max(0.0, SEGUNDOS_SIN_DETECCION - tiempo_en_estado)
            texto = f"Falta: {', '.join(faltantes)} ({restante:.1f}s)"
            color = (0, 165, 255)
        else:
            # Todo completo: no se genera reporte ni alerta
            texto = "Todo detectado (casco, chaleco, senal)"
            color = (0, 200, 0)

        annotated_frame = results[0].plot()
        cv2.putText(annotated_frame, texto, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Seguridad Vial - DroidCam", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r") and len(faltantes) > 0:
            generar_reporte(fuente_activa, ahora - inicio_estado, faltantes, detectados_estables)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()