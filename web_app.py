import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import cv2

from detector import (
    BASE_DIR,
    CONF_THRESHOLD,
    DROIDCAM_SOURCE,
    MIN_CONFIDENCE_POR_CLASE,
    MIN_DETECTIONS_CONSECUTIVAS_POR_CLASE,
    SEGUNDOS_SIN_DETECCION,
    clasificar_elemento,
    generar_reporte,
    identificar_faltantes,
)


class DetectorWeb:
    def __init__(self):
        self.model = None
        self.capture = None
        self.source = "Inicializando..."
        self.lock = threading.Lock()
        self.latest_frame = None
        self.last_status = {
            "fuente": self.source,
            "detectados": [],
            "faltantes": ["casco", "chaleco", "senal"],
            "segundos": 0.0,
            "reporte_enviado": False,
            "error": None,
        }
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._detect_loop, daemon=True)

    def start(self):
        self.worker.start()

    @staticmethod
    def _open_camera():
        capture = cv2.VideoCapture(DROIDCAM_SOURCE)
        if capture.isOpened():
            return capture, DROIDCAM_SOURCE

        capture.release()
        for index in (0, 1, 2):
            capture = cv2.VideoCapture(index)
            if capture.isOpened():
                print(f"[INFO] Usando cámara local o virtual con índice {index}.", flush=True)
                return capture, f"camara_{index}"
            capture.release()

        raise RuntimeError(
            "No se encontró una cámara disponible. Inicia DroidCam o conecta una cámara local."
        )

    def _detect_loop(self):
        ultimo_estado_faltantes = None
        inicio_estado = time.time()
        reporte_enviado = False
        apariciones_consecutivas = {"casco": 0, "chaleco": 0, "senal": 0}

        try:
            from ultralytics import YOLO

            print("[INFO] Cargando modelo YOLO...", flush=True)
            self.model = YOLO(str(BASE_DIR / "best.pt"))
            self.capture, self.source = self._open_camera()
            with self.lock:
                self.last_status["fuente"] = self.source

            while not self.stop_event.is_set():
                ret, frame = self.capture.read()
                if not ret or frame is None:
                    time.sleep(0.3)
                    continue

                results = self.model.predict(
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
                    for index, box in enumerate(boxes):
                        clase = self.model.names[int(box.cls[0])]
                        confianza = float(box.conf[0])
                        elemento = clasificar_elemento(clase)
                        if elemento is not None and confianza >= MIN_CONFIDENCE_POR_CLASE[elemento]:
                            detectados_raw.add(elemento)
                            indices_validos.append(index)
                    results[0].boxes = boxes[indices_validos]

                detectados_estables = set()
                for elemento in ("casco", "chaleco", "senal"):
                    if elemento in detectados_raw:
                        apariciones_consecutivas[elemento] += 1
                    else:
                        apariciones_consecutivas[elemento] = 0
                    if (
                        apariciones_consecutivas[elemento]
                        >= MIN_DETECTIONS_CONSECUTIVAS_POR_CLASE[elemento]
                    ):
                        detectados_estables.add(elemento)

                faltantes = identificar_faltantes(detectados_estables)
                ahora = time.time()
                if faltantes != ultimo_estado_faltantes:
                    ultimo_estado_faltantes = faltantes
                    inicio_estado = ahora
                    reporte_enviado = False

                tiempo_en_estado = ahora - inicio_estado
                if (
                    faltantes
                    and tiempo_en_estado >= SEGUNDOS_SIN_DETECCION
                    and not reporte_enviado
                ):
                    generar_reporte(self.source, tiempo_en_estado, faltantes, detectados_estables)
                    reporte_enviado = True

                if faltantes:
                    restante = max(0.0, SEGUNDOS_SIN_DETECCION - tiempo_en_estado)
                    texto = f"Falta: {', '.join(faltantes)} ({restante:.1f}s)"
                    color = (0, 165, 255)
                else:
                    texto = "Todo detectado (casco, chaleco, senal)"
                    color = (0, 200, 0)

                annotated_frame = results[0].plot()
                cv2.putText(
                    annotated_frame,
                    texto,
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )
                encoded, buffer = cv2.imencode(".jpg", annotated_frame)
                if not encoded:
                    raise RuntimeError("No se pudo codificar el fotograma de la cámara.")

                with self.lock:
                    self.latest_frame = buffer.tobytes()
                    self.last_status = {
                        "fuente": self.source,
                        "detectados": sorted(detectados_estables),
                        "faltantes": faltantes,
                        "segundos": round(tiempo_en_estado, 1),
                        "reporte_enviado": reporte_enviado,
                        "error": None,
                    }
        except Exception as error:
            with self.lock:
                self.last_status["error"] = str(error)
            print(f"[ERROR] Falló el detector: {error}", flush=True)
            raise
        finally:
            if self.capture is not None:
                self.capture.release()

    def snapshot(self):
        with self.lock:
            return dict(self.last_status)

    def frame(self):
        with self.lock:
            return self.latest_frame

    def report_now(self):
        with self.lock:
            status = dict(self.last_status)
            if not status["faltantes"]:
                return False
            generar_reporte(
                self.source,
                status["segundos"],
                status["faltantes"],
                set(status["detectados"]),
            )
            self.last_status["reporte_enviado"] = True
            return True


class WebHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        detector = self.server.detector

        if path == "/":
            page = (BASE_DIR / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return

        if path == "/api/status":
            self._send_json(200, detector.snapshot())
            return

        if path == "/video":
            frame = detector.frame()
            while frame is None:
                if detector.snapshot()["error"]:
                    self.send_error(503, "El detector no pudo iniciar.")
                    return
                time.sleep(0.1)
                frame = detector.frame()

            self.send_response(200)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    frame = detector.frame()
                    if frame is None:
                        time.sleep(0.1)
                        continue
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                    )
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                return

        self.send_error(404, "Ruta no encontrada.")

    def do_POST(self):
        if urlsplit(self.path).path != "/api/report":
            self.send_error(404, "Ruta no encontrada.")
            return

        if self.server.detector.report_now():
            self._send_json(200, {"mensaje": "Reporte guardado correctamente."})
        else:
            self._send_json(409, {"mensaje": "No hay elementos faltantes para reportar."})

    def log_message(self, format_string, *args):
        print(f"[HTTP] {self.address_string()} - {format_string % args}")


def main():
    detector = DetectorWeb()
    server = ThreadingHTTPServer(("127.0.0.1", 8000), WebHandler)
    server.detector = detector
    detector.start()
    print("[INFO] Abre http://127.0.0.1:8000 en tu navegador.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Cerrando servidor...", flush=True)
    finally:
        detector.stop_event.set()
        server.server_close()


if __name__ == "__main__":
    main()
