from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Ruta absoluta hacia models/best.pt relativa a la raíz del proyecto
RUTA_MODELO = Path(__file__).resolve().parent.parent / "models" / "best.pt"
_modelo_yolo = None


def _obtener_modelo_yolo() -> YOLO:
    """Carga el modelo YOLOv8-seg una sola vez en memoria."""
    global _modelo_yolo
    if _modelo_yolo is None:
        if not RUTA_MODELO.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo entrenado en: {RUTA_MODELO}"
            )
        _modelo_yolo = YOLO(str(RUTA_MODELO))
    return _modelo_yolo


def analizar_cuadro(img_gris: np.ndarray) -> dict:
    """
    Etapa 2 (Acondicionamiento): CLAHE + Suavizado Gaussiano.
    Etapa 3 (Extracción geométrica y topográfica): Otsu, fitEllipse,
    momentos espaciales, ovalamiento, excentricidad, rugosidad y mapa Sobel.
    """
    # --- ETAPA 2: Acondicionamiento de señal ---
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gris)
    img_suave = cv2.GaussianBlur(img_clahe, (5, 5), 0)

    # --- ETAPA 3: Segmentación automática (Otsu) ---
    _, mascara = cv2.threshold(
        img_suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Limpieza morfológica básica para eliminar ruido aislado
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)

    contornos, _ = cv2.findContours(
        mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Valores por defecto si no se detecta un contorno válido
    excentricidad = 0.0
    ovalamiento = 0.0
    rugosidad = 0.0
    area = 0.0
    img_contorno = cv2.cvtColor(img_suave, cv2.COLOR_GRAY2BGR)

    # Gradientes Sobel (para rugosidad y mapa de relieve/hillshade)
    sobel_x = cv2.Sobel(img_suave, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_suave, cv2.CV_64F, 0, 1, ksize=3)
    magnitud_sobel = np.sqrt(sobel_x**2 + sobel_y**2)

    # Hillshade simple (simulación de relieve iluminado)
    hillshade = np.clip(128 + 0.6 * sobel_x + 0.6 * sobel_y, 0, 255).astype(np.uint8)

    if contornos:
        contorno_principal = max(contornos, key=cv2.contourArea)
        area = float(cv2.contourArea(contorno_principal))

        # Momentos espaciales para el centroide real
        momentos = cv2.moments(contorno_principal)
        if momentos["m00"] != 0:
            cx = momentos["m10"] / momentos["m00"]
            cy = momentos["m01"] / momentos["m00"]
        else:
            cx, cy = 128.0, 128.0

        # Excentricidad: distancia normalizada respecto al centro ideal (128, 128)
        excentricidad = float(np.hypot(cx - 128.0, cy - 128.0) / 128.0)

        # Ajuste geométrico de elipse (fitEllipse requiere al menos 5 puntos)
        if len(contorno_principal) >= 5:
            elipse = cv2.fitEllipse(contorno_principal)
            (_, _), (eje_1, eje_2), _ = elipse
            eje_mayor = max(eje_1, eje_2)
            eje_menor = min(eje_1, eje_2)
            ovalamiento = float((eje_mayor - eje_menor) / max(eje_mayor, 1e-6))

            # Dibuja elipse ajustada (verde) y referencia circular ideal (azul)
            cv2.ellipse(img_contorno, elipse, (0, 255, 0), 2)
            cv2.circle(img_contorno, (128, 128), 70, (255, 180, 0), 1)
            cv2.circle(img_contorno, (int(cx), int(cy)), 3, (0, 0, 255), -1)

        # Rugosidad superficial: desviación del gradiente Sobel dentro del tapón
        mascara_interna = np.zeros_like(mascara)
        cv2.drawContours(mascara_interna, [contorno_principal], -1, 255, -1)
        # Erosionamos ligeramente para medir la superficie interna y no el borde externo
        mascara_interna = cv2.erode(mascara_interna, kernel, iterations=2)
        pixeles_internos = magnitud_sobel[mascara_interna == 255]
        if pixeles_internos.size > 0:
            rugosidad = float(np.mean(pixeles_internos))

    return {
        "excentricidad": round(excentricidad, 4),
        "ovalamiento": round(ovalamiento, 4),
        "rugosidad": round(rugosidad, 2),
        "area_px": round(area, 1),
        "img_clahe": img_clahe,
        "img_suave": img_suave,
        "mascara": mascara,
        "img_contorno": img_contorno,
        "hillshade": hillshade,
        "magnitud_sobel": magnitud_sobel,
    }


def analizar_imagen_real(
    img_bgr: np.ndarray, sensibilidad: float = 1.0, umbral_critico: int = 95
) -> dict:
    """
    Analiza la superficie real del tapón usando YOLOv8-seg (models/best.pt):
    - Aísla exclusivamente la silueta del tapón ignorando fondo, reflejos y rodillos.
    - Devuelve hay_tapon=False al instante en cuadros vacíos (Null).
    - Analiza abolladuras y relieve únicamente dentro de la máscara segmentada.
    """
    if img_bgr is None or getattr(img_bgr, "size", 0) == 0:
        gris = np.zeros((1, 1), dtype=np.uint8)
        vacio_bgr = np.zeros((1, 1, 3), dtype=np.uint8)
        return {
            "hay_tapon": False,
            "sin_gotas": gris,
            "heatmap_bgr": vacio_bgr,
            "superpuesta_bgr": vacio_bgr,
            "indice_rugosidad": 0.0,
            "pct_zona_critica": 0.0,
        }

    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Segmentación del cuerpo del tapón con YOLOv8-seg
    modelo = _obtener_modelo_yolo()
    resultados = modelo.predict(source=img_bgr, imgsz=640, conf=0.45, verbose=False)
    res = resultados[0]

    mascara_tapon = np.zeros_like(gris)
    hay_deteccion = False

    if res.masks is not None and len(res.masks) > 0:
        confs = (
            res.boxes.conf.cpu().numpy()
            if res.boxes is not None
            else np.ones(len(res.masks))
        )
        idx_mejor = int(np.argmax(confs))

        # masks.xy entrega el polígono ya escalado al tamaño original de img_bgr
        poligono = res.masks.xy[idx_mejor]
        if poligono is not None and len(poligono) >= 3:
            pts = np.round(poligono).astype(np.int32)
            cv2.fillPoly(mascara_tapon, [pts], 255)
        else:
            mask_raw = res.masks.data[idx_mejor].cpu().numpy()
            mask_resized = cv2.resize(
                mask_raw,
                (gris.shape[1], gris.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            mascara_tapon = (mask_resized > 0.5).astype(np.uint8) * 255

        if np.count_nonzero(mascara_tapon) > 0.02 * gris.size:
            hay_deteccion = True

    # Protección para cuadros sin tapón (clase Null / fondo vacío)
    if not hay_deteccion:
        return {
            "hay_tapon": False,
            "sin_gotas": gris,
            "heatmap_bgr": img_bgr.copy(),
            "superpuesta_bgr": img_bgr.copy(),
            "indice_rugosidad": 0.0,
            "pct_zona_critica": 0.0,
        }

    # --- FILTRO 1: Erosión suave sobre la silueta IA para evitar falso gradiente en el borde ---
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mascara_interna = cv2.erode(mascara_tapon, kernel_erode, iterations=1)
    mascara_valida = (mascara_interna == 255) & (gris < 245)

    # 2. Suavizado inteligente + CLAHE suave
    sin_ruido = cv2.bilateralFilter(gris, d=7, sigmaColor=35, sigmaSpace=35)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    mejorada = clahe.apply(sin_ruido)

    # 3. Extracción de textura + detección de hundimientos amplios (abolladuras)
    kernel_tex = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    blackhat = cv2.morphologyEx(mejorada, cv2.MORPH_BLACKHAT, kernel_tex)
    tophat = cv2.morphologyEx(mejorada, cv2.MORPH_TOPHAT, kernel_tex)

    # Detector de zonas hundidas/oscuras amplias (abolladuras grandes)
    fondo_iluminacion = cv2.GaussianBlur(mejorada, (51, 51), 0)
    hundimiento = cv2.subtract(fondo_iluminacion, mejorada)
    hundimiento_limpio = np.clip(hundimiento.astype(np.float32) - 12.0, 0, 255)

    sobel_x = cv2.Sobel(mejorada, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(mejorada, cv2.CV_64F, 0, 1, ksize=3)
    gradiente = np.sqrt(sobel_x**2 + sobel_y**2)

    # A) Mapa combinado: bordes + grietas + interior de abolladuras
    grad_limpio = np.clip(gradiente - 18.0, 0, 255)
    morf_limpia = np.clip(
        (blackhat.astype(np.float32) + tophat.astype(np.float32)) - 10.0, 0, 255
    )
    mapa_fallas = (
        np.clip(grad_limpio * 0.6, 0, 255) * 0.30
        + np.clip(morf_limpia * 2.0, 0, 255) * 0.35
        + np.clip(hundimiento_limpio * 3.0, 0, 255) * 0.35
    )
    mapa_fallas = np.clip(mapa_fallas * sensibilidad, 0, 255).astype(np.uint8)
    mapa_fallas[~mascara_valida] = 0

    # --- FILTRO 2: Limpiar ruido, fusionar y RELLENAR abolladuras con profundidad ---
    zona_critica = (mapa_fallas > umbral_critico).astype(np.uint8) * 255

    # 1) Borrar puntitos sueltos
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    zona_limpia = cv2.morphologyEx(zona_critica, cv2.MORPH_OPEN, kernel_open)

    # 2) Unir bordes rotos y cerrar huecos de la misma abolladura
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))
    zona_cerrada = cv2.morphologyEx(zona_limpia, cv2.MORPH_CLOSE, kernel_close)

    # 3) Rellenar contornos sólidos que superen el área mínima
    contornos_def, _ = cv2.findContours(
        zona_cerrada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    mascara_defectos = np.zeros_like(zona_critica)
    AREA_MIN_MANCHA = 120

    for cnt in contornos_def:
        if cv2.contourArea(cnt) >= AREA_MIN_MANCHA:
            cv2.drawContours(mascara_defectos, [cnt], -1, 255, thickness=cv2.FILLED)

    mascara_defectos[~mascara_valida] = 0

    # B) Mapa visual: forma 3D azul/cian + abolladuras rellenas manteniendo profundidad
    dist = cv2.distanceTransform(mascara_defectos, cv2.DIST_L2, 5)
    if dist.max() > 0:
        dist_norm = (dist / dist.max()) * 255.0
    else:
        dist_norm = np.zeros_like(dist)

    profundidad_defecto = np.clip(
        155.0
        + 0.35 * dist_norm
        + 0.35 * mapa_fallas.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)

    forma_base = (mejorada.astype(np.float32) / 255.0 * 75.0 + 10.0).astype(np.uint8)
    mapa_visual = forma_base.copy()
    idx_def = mascara_defectos > 0
    mapa_visual[idx_def] = np.maximum(forma_base[idx_def], profundidad_defecto[idx_def])

    heatmap_bgr = cv2.applyColorMap(mapa_visual, cv2.COLORMAP_TURBO)
    sin_gotas = cv2.cvtColor(mejorada, cv2.COLOR_GRAY2BGR)
    superpuesta_bgr = cv2.addWeighted(img_bgr, 0.55, heatmap_bgr, 0.45, 0)

    # Dibuja el contorno exacto detectado por YOLOv8-seg en cian sobre la imagen superpuesta
    contornos_ia, _ = cv2.findContours(
        mascara_tapon, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(superpuesta_bgr, contornos_ia, -1, (255, 255, 0), 2)

    pixeles_validos = int(np.count_nonzero(mascara_valida))
    pixeles_defectos = int(np.count_nonzero(mascara_defectos))
    pct_zona_critica = (
        100.0 * pixeles_defectos / pixeles_validos if pixeles_validos > 0 else 0.0
    )
    valores_validos = mapa_fallas[mascara_valida]
    indice_rugosidad = (
        float(np.mean(valores_validos)) if valores_validos.size > 0 else 0.0
    )

    return {
        "hay_tapon": True,
        "sin_gotas": sin_gotas,
        "heatmap_bgr": heatmap_bgr,
        "superpuesta_bgr": superpuesta_bgr,
        "indice_rugosidad": round(indice_rugosidad, 2),
        "pct_zona_critica": round(pct_zona_critica, 2),
    }


if __name__ == "__main__":
    img = cv2.imread("data/muestra_tapon.png", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: primero ejecuta python src/simulador.py")
    else:
        res = analizar_cuadro(img)
        cv2.imwrite("data/muestra_contorno.png", res["img_contorno"])
        cv2.imwrite("data/muestra_hillshade.png", res["hillshade"])
        print("Métricas extraídas de data/muestra_tapon.png:")
        print(f"  Excentricidad : {res['excentricidad']}")
        print(f"  Ovalamiento   : {res['ovalamiento']}")
        print(f"  Rugosidad     : {res['rugosidad']}")
        print(f"  Área (px)     : {res['area_px']}")
        print("Imágenes guardadas en data/muestra_contorno.png y data/muestra_hillshade.png")