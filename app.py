import io
import os
import re
import time
from datetime import datetime
import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.simulador import generar_cuadro_tapon
from src.vision import analizar_cuadro, analizar_imagen_real
from src.modelo import FEATURES

st.set_page_config(
    page_title="Monitoreo Predictivo de Tapón Perforador",
    layout="wide",
)

CARPETA_VIVO = "data/capturas_vivo"
RUTA_APROBADOS = "data/registro_aprobados.csv"
RUTA_CUARENTENA = "data/registro_cuarentena.csv"
RUTA_HISTORIAL_FRAMES = "data/historial_frames.csv"

os.makedirs(CARPETA_VIVO, exist_ok=True)
os.makedirs("data", exist_ok=True)

st.title("Sistema de Visión y Analítica Predictiva — Tapón Perforador")
st.caption("Control de Calidad en Tiempo Real | Detección YOLOv8-seg & Interlock de Línea")


@st.cache_data
def cargar_datos():
    return pd.read_csv("data/dataset_completo.csv")


@st.cache_resource
def cargar_modelos():
    clf = joblib.load("data/modelo_riesgo.joblib")
    reg = joblib.load("data/modelo_rul.joblib")
    return clf, reg


def incrementar_id_tapon(codigo: str) -> str:
    codigo = codigo.strip() or "TP-001"
    matches = list(re.finditer(r"\d+", codigo))
    if not matches:
        return f"{codigo}-2"
    ultimo = matches[-1]
    num_str = ultimo.group()
    nuevo_num = str(int(num_str) + 1).zfill(len(num_str))
    return codigo[: ultimo.start()] + nuevo_num + codigo[ultimo.end() :]


def cargar_csv_seguro(ruta: str, columnas: list[str]) -> pd.DataFrame:
    if os.path.exists(ruta):
        return pd.read_csv(ruta)
    return pd.DataFrame(columns=columnas)


def guardar_fila_csv(ruta: str, registro: dict, columnas: list[str]) -> pd.DataFrame:
    df_actual = cargar_csv_seguro(ruta, columnas)
    df_nuevo = pd.DataFrame([registro])
    df_final = (
        df_nuevo
        if df_actual.empty
        else pd.concat([df_actual, df_nuevo], ignore_index=True)
    )
    df_final.to_csv(ruta, index=False)
    return df_final


def anexar_frames_historial(
    df_seq: pd.DataFrame,
    lote: str,
    id_tapon: str,
    turno: str,
    dictamen: str,
    fecha_hora: str,
):
    cols_frames = [
        "Fecha_Hora",
        "Lote_Orden",
        "ID_Tapon",
        "Turno",
        "Dictamen_Secuencia",
        "Frame",
        "Tapón",
        "Rugosidad",
        "Área Crítica (%)",
        "Latencia (ms)",
        "Estado",
        "Racha Alerta",
    ]
    df_copia = df_seq.copy()
    df_copia.insert(0, "Dictamen_Secuencia", dictamen)
    df_copia.insert(0, "Turno", turno)
    df_copia.insert(0, "ID_Tapon", id_tapon)
    df_copia.insert(0, "Lote_Orden", lote)
    df_copia.insert(0, "Fecha_Hora", fecha_hora)

    df_hist = cargar_csv_seguro(RUTA_HISTORIAL_FRAMES, cols_frames)
    df_final = (
        df_copia
        if df_hist.empty
        else pd.concat([df_hist, df_copia], ignore_index=True)
    )
    df_final.to_csv(RUTA_HISTORIAL_FRAMES, index=False)


def redimensionar_preview(img_bgr: np.ndarray, ancho_max: int = 480) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if w <= ancho_max:
        return img_bgr
    nuevo_h = max(1, int(h * (ancho_max / w)))
    return cv2.resize(img_bgr, (ancho_max, nuevo_h), interpolation=cv2.INTER_AREA)


def dibujar_grafica_hmi(
    nombres: list[str],
    valores: list[float],
    umbral_alerta: float,
    umbral_prec: float,
    total_esperado: int,
) -> np.ndarray:
    """Renderiza la gráfica de tendencia en vivo usando OpenCV (< 2 ms por frame)."""
    W, H = 960, 210
    left, right, top, bottom = 68, 24, 26, 34
    plot_w = W - left - right
    plot_h = H - top - bottom

    canvas = np.full((H, W, 3), (250, 251, 252), dtype=np.uint8)
    cv2.rectangle(
        canvas, (left, top), (left + plot_w, top + plot_h), (210, 214, 220), 1
    )

    max_val = max(valores) if valores else 0.0
    max_y = max(umbral_alerta * 1.45, max_val * 1.18, 5.0)

    def y_px(v: float) -> int:
        return int(top + plot_h - (v / max_y) * plot_h)

    def x_px(idx: int) -> int:
        if total_esperado <= 1:
            return left + plot_w // 2
        return int(left + (idx / (total_esperado - 1)) * plot_w)

    for frac in (0.0, 0.5, 1.0):
        val_g = max_y * frac
        yp = y_px(val_g)
        cv2.line(canvas, (left, yp), (left + plot_w, yp), (232, 235, 238), 1)
        cv2.putText(
            canvas,
            f"{val_g:.1f}%",
            (10, yp + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (90, 95, 105),
            1,
            cv2.LINE_AA,
        )

    y_prec = y_px(umbral_prec)
    for x in range(left, left + plot_w, 12):
        cv2.line(
            canvas,
            (x, y_prec),
            (min(x + 6, left + plot_w), y_prec),
            (0, 155, 245),
            1,
        )

    y_alerta = y_px(umbral_alerta)
    for x in range(left, left + plot_w, 14):
        cv2.line(
            canvas,
            (x, y_alerta),
            (min(x + 9, left + plot_w), y_alerta),
            (40, 40, 220),
            2,
        )
    cv2.putText(
        canvas,
        f"Umbral Paro ({umbral_alerta:.1f}%)",
        (left + 8, max(top + 14, y_alerta - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (40, 40, 220),
        1,
        cv2.LINE_AA,
    )

    puntos = [(x_px(i), y_px(v)) for i, v in enumerate(valores)]
    for i in range(1, len(puntos)):
        cv2.line(
            canvas, puntos[i - 1], puntos[i], (60, 60, 190), 2, cv2.LINE_AA
        )

    for i, (px, py) in enumerate(puntos):
        v = valores[i]
        if v >= umbral_alerta:
            color = (35, 35, 225)
        elif v >= umbral_prec:
            color = (0, 145, 245)
        else:
            color = (50, 165, 65)
        cv2.circle(canvas, (px, py), 4, color, -1, cv2.LINE_AA)

    if nombres:
        etiqueta_izq = f"Inicio: {nombres[0]}"
        etiqueta_der = f"Actual: {nombres[-1]} ({valores[-1]:.1f}%)"
        cv2.putText(
            canvas,
            etiqueta_izq,
            (left, H - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (90, 95, 105),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            etiqueta_der,
            (left + plot_w - 230, H - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (40, 40, 50),
            1,
            cv2.LINE_AA,
        )

    return canvas


def generar_reporte_excel(
    df_aprobados: pd.DataFrame,
    df_cuarentena: pd.DataFrame,
    df_hist_frames: pd.DataFrame,
    df_seq_activa: pd.DataFrame | None,
    resumen_meta: dict,
) -> bytes | None:
    try:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_meta = pd.DataFrame(
                [
                    {"Parámetro / Indicador": k, "Valor": v}
                    for k, v in resumen_meta.items()
                ]
            )
            df_meta.to_excel(writer, sheet_name="Resumen_Operativo", index=False)

            if not df_aprobados.empty:
                df_aprobados.to_excel(
                    writer, sheet_name="Tapones_Aprobados", index=False
                )

            if not df_cuarentena.empty:
                df_cuarentena.to_excel(
                    writer, sheet_name="Revision_y_Chatarreria", index=False
                )

            if not df_hist_frames.empty:
                df_hist_frames.to_excel(
                    writer, sheet_name="Historial_Frames", index=False
                )
            elif df_seq_activa is not None and not df_seq_activa.empty:
                df_seq_activa.to_excel(
                    writer, sheet_name="Frames_Secuencia_Activa", index=False
                )

            for sheet in writer.sheets.values():
                for col in sheet.columns:
                    max_len = max(
                        len(str(cell.value)) if cell.value is not None else 0
                        for cell in col
                    )
                    sheet.column_dimensions[col[0].column_letter].width = min(
                        max_len + 4, 45
                    )

        return buffer.getvalue()
    except ImportError:
        return None


COLS_APROBADOS = [
    "Fecha_Hora",
    "Lote_Orden",
    "ID_Tapon",
    "Turno",
    "Dictamen",
    "Frames_Analizados",
    "Frames_Precaucion",
    "Frames_Alerta_Aislada",
    "Area_Critica_Max_Pct",
    "Area_Critica_Prom_Pct",
    "Rugosidad_Max",
    "Observaciones",
]

COLS_CUARENTENA = [
    "Fecha_Hora",
    "Lote_Orden",
    "ID_Tapon",
    "Turno",
    "Disposicion",
    "Tipo_Paro",
    "Frames_Analizados",
    "Frames_Alerta",
    "Max_Racha_Consecutiva",
    "Area_Critica_Max_Pct",
    "Rugosidad_Max",
    "Frame_Critico",
    "Motivo",
]

df = cargar_datos()
modelo_riesgo, modelo_rul = cargar_modelos()

tab1, tab2, tab3 = st.tabs(
    [
        "1. Simulación de Corrida (Predictivo)",
        "2. Inspección y Control de Producción (Planta)",
        "3. Stream en Vivo (Simulador HMI)",
    ]
)

# ==============================================================================
# PESTAÑA 1: SIMULACIÓN DE CORRIDA COMPLETA
# ==============================================================================
with tab1:
    st.sidebar.header("Control de Corrida (Edge)")
    corridas = sorted(df["corrida_id"].unique())
    corrida_sel = st.sidebar.selectbox("Seleccionar Corrida ID", corridas)

    df_corrida = df[df["corrida_id"] == corrida_sel].reset_index(drop=True)
    ciclo_sel = st.sidebar.slider(
        "Ciclo de operación (pieza actual)",
        min_value=int(df_corrida["ciclo"].min()),
        max_value=int(df_corrida["ciclo"].max()),
        value=int(df_corrida["ciclo"].max() // 2),
    )

    fila = df_corrida[df_corrida["ciclo"] == ciclo_sel].iloc[0]
    idx_global = int(
        df[(df["corrida_id"] == corrida_sel) & (df["ciclo"] == ciclo_sel)].index[0]
    )

    t_inicio = time.perf_counter()
    img_cruda = generar_cuadro_tapon(
        desgaste=float(fila["desgaste_real"]), seed=idx_global
    )
    vis = analizar_cuadro(img_cruda)

    X_actual = pd.DataFrame([fila[FEATURES]])
    prob_riesgo = float(modelo_riesgo.predict_proba(X_actual)[0, 1]) * 100.0
    rul_estimado = max(0.0, float(modelo_rul.predict(X_actual)[0]))
    tiempo_ms = (time.perf_counter() - t_inicio) * 1000.0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Riesgo de Defecto", f"{prob_riesgo:.1f}%")
    col2.metric("Vida Útil Restante (RUL)", f"{rul_estimado:.1f} ciclos")
    col3.metric("Temperatura", f"{fila['temperatura_c']:.1f} °C")
    col4.metric("Carga de Motor", f"{fila['carga_motor_pct']:.1f} %")
    col5.metric("Latencia Edge", f"{tiempo_ms:.1f} ms", delta="Ventana: 5000 ms")

    factores = []
    if vis["ovalamiento"] > 0.14:
        factores.append(f"ovalamiento elevado ({vis['ovalamiento']:.3f})")
    if vis["rugosidad"] > 72:
        factores.append(f"rugosidad superficial crítica ({vis['rugosidad']:.1f})")
    if fila["temperatura_c"] > 995:
        factores.append(f"sobrecalentamiento ({fila['temperatura_c']:.1f} °C)")
    if vis["excentricidad"] > 0.05:
        factores.append(f"descentrado/excentricidad ({vis['excentricidad']:.3f})")

    explicacion = (
        ", ".join(factores)
        if factores
        else "parámetros geométricos y térmicos dentro de rango nominal."
    )

    if prob_riesgo >= 70 or rul_estimado <= 5:
        st.error(
            f"**ALERTA CRÍTICA — Programar cambio de tapón:** Riesgo alto ({prob_riesgo:.1f}%) impulsado por {explicacion}"
        )
    elif prob_riesgo >= 35:
        st.warning(
            f"**PRECAUCIÓN — Monitoreo estrecho:** Desgaste progresivo detectado ({prob_riesgo:.1f}%) por {explicacion}"
        )
    else:
        st.success(
            f"**OPERACIÓN NORMAL:** Riesgo bajo ({prob_riesgo:.1f}%) — {explicacion}"
        )

    st.divider()

    st.subheader("Diagnóstico Visual y Topográfico del Tapón")
    v1, v2, v3, v4 = st.columns(4)

    with v1:
        st.image(
            img_cruda,
            caption="1. Entrada cruda (baja res. + ruido)",
            use_container_width=True,
        )
    with v2:
        st.image(
            vis["img_suave"],
            caption="2. Acondicionada (CLAHE + Gauss)",
            use_container_width=True,
        )
    with v3:
        st.image(
            vis["img_contorno"],
            channels="BGR",
            caption="3. Otsu + fitEllipse vs Ideal",
            use_container_width=True,
        )
    with v4:
        fig_iso, ax_iso = plt.subplots(figsize=(3.5, 3.5))
        ax_iso.imshow(vis["hillshade"], cmap="turbo")
        ax_iso.contour(
            vis["img_suave"], levels=7, colors="black", linewidths=0.6, alpha=0.75
        )
        ax_iso.axis("off")
        st.pyplot(fig_iso)
        plt.close(fig_iso)
        st.caption("4. Mapa de relieve (Hillshade + Isolíneas)")

    st.subheader(f"Tendencias Operativas — Corrida #{corrida_sel}")
    t1, t2 = st.columns(2)

    with t1:
        fig1, ax1 = plt.subplots(figsize=(5, 2.8))
        ax1.plot(
            df_corrida["ciclo"],
            df_corrida["ovalamiento"],
            label="Ovalamiento",
            alpha=0.5,
        )
        ax1.plot(
            df_corrida["ciclo"],
            df_corrida["ovalamiento_ma3"],
            label="Tendencia móvil (MA3)",
            linewidth=2,
        )
        ax1.axvline(ciclo_sel, color="red", linestyle="--", label="Ciclo actual")
        ax1.set_xlabel("Ciclo")
        ax1.set_ylabel("Ovalamiento")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)
        st.pyplot(fig1)
        plt.close(fig1)

    with t2:
        fig2, ax2 = plt.subplots(figsize=(5, 2.8))
        ax2.plot(
            df_corrida["ciclo"],
            df_corrida["rugosidad"],
            label="Rugosidad (Sobel)",
            color="orange",
            alpha=0.5,
        )
        ax2.plot(
            df_corrida["ciclo"],
            df_corrida["rugosidad_ma3"],
            label="Tendencia móvil (MA3)",
            color="darkorange",
            linewidth=2,
        )
        ax2.axvline(ciclo_sel, color="red", linestyle="--", label="Ciclo actual")
        ax2.set_xlabel("Ciclo")
        ax2.set_ylabel("Rugosidad")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)
        plt.close(fig2)

    st.divider()

    st.subheader("Registro del Operador y Recalibración (Human-in-the-Loop)")
    ruta_log = "data/historial_operador.csv"

    with st.form("form_pdca"):
        c_dec, c_nota = st.columns([1, 2])
        decision = c_dec.selectbox(
            "Acción tomada",
            [
                "Continuar corrida",
                "Inspección visual preventiva",
                "Cambio de tapón programado",
                "Falsa alarma (recalibrar)",
            ],
        )
        nota = c_nota.text_input("Observaciones del operador (opcional)")
        enviar = st.form_submit_button("Guardar en historial PDCA")

        if enviar:
            nuevo = pd.DataFrame(
                [
                    {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "corrida_id": corrida_sel,
                        "ciclo": ciclo_sel,
                        "riesgo_pct": round(prob_riesgo, 1),
                        "rul_estimado": round(rul_estimado, 1),
                        "decision": decision,
                        "nota": nota,
                    }
                ]
            )
            if os.path.exists(ruta_log):
                nuevo.to_csv(ruta_log, mode="a", header=False, index=False)
            else:
                nuevo.to_csv(ruta_log, index=False)
            st.success("Decisión registrada en data/historial_operador.csv")

    if os.path.exists(ruta_log):
        st.dataframe(pd.read_csv(ruta_log).tail(5), use_container_width=True)


# ==============================================================================
# PESTAÑA 2: INSPECCIÓN CONSECUTIVA, APROBACIÓN Y CONTROL DE PARO
# ==============================================================================
with tab2:
    if "archivos_guardados_ids" not in st.session_state:
        st.session_state["archivos_guardados_ids"] = set()
    if "cache_analisis_vivo" not in st.session_state:
        st.session_state["cache_analisis_vivo"] = {}
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0
    if "ultimo_evento_info" not in st.session_state:
        st.session_state["ultimo_evento_info"] = None
    if st.session_state.pop("reset_slider_planta", False):
        st.session_state.pop("slider_frame_planta", None)
    if "pending_tapon_id" in st.session_state:
        st.session_state["input_tapon_id"] = st.session_state.pop("pending_tapon_id")
    elif "input_tapon_id" not in st.session_state:
        st.session_state["input_tapon_id"] = "TP-001"

    def limpiar_area_activa():
        for nombre in os.listdir(CARPETA_VIVO):
            ruta_f = os.path.join(CARPETA_VIVO, nombre)
            if os.path.isfile(ruta_f):
                os.remove(ruta_f)
        st.session_state["archivos_guardados_ids"].clear()
        st.session_state["cache_analisis_vivo"].clear()
        st.session_state["uploader_key"] += 1
        st.session_state.pop("clave_secuencia_vista", None)
        st.session_state["reset_slider_planta"] = True

    st.subheader("Control de Calidad en Línea — Inspección Consecutiva de Tapones")

    tr1, tr2, tr3 = st.columns([1.5, 1.5, 1])
    lote_id = tr1.text_input("Lote / Orden de Producción", value="ORD-2026-041")
    tapon_id = tr2.text_input(
        "ID de Tapón Perforador (siguiente pieza)",
        key="input_tapon_id",
    )
    turno_sel = tr3.selectbox(
        "Turno", ["Turno 1 (Matutino)", "Turno 2 (Vespertino)", "Turno 3 (Nocturno)"]
    )

    col_up, col_sync, col_clear = st.columns([3.4, 1.1, 1.1])
    with col_up:
        archivos = st.file_uploader(
            "Cargar secuencia del tapón actual o capturar desde cámara móvil (.jpg, .png)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key=f"uploader_vivo_{st.session_state['uploader_key']}",
        )
    with col_sync:
        st.write("")
        st.write("")
        if st.button(
            "🔄 Sincronizar",
            use_container_width=True,
            help="Actualizar capturas enviadas desde celular",
        ):
            st.rerun()
    with col_clear:
        st.write("")
        st.write("")
        if st.button(
            "🗑️ Descartar",
            use_container_width=True,
            help="Limpiar imágenes actuales sin guardar en base de datos",
        ):
            limpiar_area_activa()
            st.session_state["ultimo_evento_info"] = None
            st.rerun()

    if archivos:
        existentes = [
            f
            for f in os.listdir(CARPETA_VIVO)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        correlativo = len(existentes) + 1

        for arch in archivos:
            id_unico = getattr(arch, "file_id", f"{arch.name}_{arch.size}")
            if id_unico in st.session_state["archivos_guardados_ids"]:
                continue

            ext = os.path.splitext(arch.name)[1].lower() or ".jpg"
            if re.search(r"frame_\d+", arch.name.lower()):
                nombre_destino = arch.name
            else:
                hora = datetime.now().strftime("%H%M%S")
                nombre_destino = f"Captura_{correlativo:03d}_{hora}{ext}"
                correlativo += 1

            ruta_destino = os.path.join(CARPETA_VIVO, nombre_destino)
            with open(ruta_destino, "wb") as f_out:
                f_out.write(arch.getbuffer())

            st.session_state["archivos_guardados_ids"].add(id_unico)
            st.session_state["ultimo_evento_info"] = None

    cal1, cal2, cal3 = st.columns(3)
    sensibilidad = cal1.slider(
        "Sensibilidad de relieve", min_value=0.5, max_value=2.0, value=1.0, step=0.1
    )
    umbral_alerta = cal2.slider(
        "Umbral de Defecto — Área Crítica (%)",
        min_value=1.0,
        max_value=30.0,
        value=8.0,
        step=0.5,
    )
    frames_disparo_req = cal3.slider(
        "Frames consecutivos para disparo de paro",
        min_value=1,
        max_value=5,
        value=2,
        help="Evita paros falsos por una salpicadura aislada; requiere N cuadros seguidos en alerta.",
    )
    umbral_precaucion = umbral_alerta * 0.5

    if st.session_state["ultimo_evento_info"]:
        ev = st.session_state["ultimo_evento_info"]
        if ev["tipo"] == "APROBADO":
            st.success(
                f"✅ **TAPÓN APROBADO Y GUARDADO EN BASE DE DATOS** | "
                f"Pieza **{ev['ID_Tapon']}** (Lote **{ev['Lote_Orden']}**) liberada con "
                f"{ev['Frames_Analizados']} frames (daño máx. {ev['Area_Critica_Max_Pct']}%). "
                f"Listo para cargar la secuencia de **{st.session_state['input_tapon_id']}**."
            )
        else:
            st.error(
                f"🛑 **LÍNEA DETENIDA — INTERLOCK PLC: STOP [0]** | "
                f"Pieza **{ev['ID_Tapon']}** (Lote **{ev['Lote_Orden']}**) enviada a "
                f"**{ev['Disposicion']}** ({ev['Frames_Alerta']} frames críticos, máx. {ev['Area_Critica_Max_Pct']}%). "
                f"Listo para inspeccionar **{st.session_state['input_tapon_id']}** tras cambio."
            )

    archivos_disco = [
        f
        for f in os.listdir(CARPETA_VIVO)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    df_seq = None

    if not archivos_disco:
        if not st.session_state["ultimo_evento_info"]:
            st.info(
                f"Línea lista (`PLC: STANDBY`) para inspeccionar **{tapon_id}**. "
                "Sube una secuencia de imágenes o captura desde cámara móvil para evaluar."
            )
    else:
        def clave_orden(nombre):
            nums = re.findall(r"\d+", nombre)
            return int(nums[0]) if nums else nombre

        archivos_ordenados = sorted(archivos_disco, key=clave_orden)

        resultados = []
        imagenes_procesadas = {}
        tiempo_total_ms = 0.0

        for nombre in archivos_ordenados:
            ruta_img = os.path.join(CARPETA_VIVO, nombre)
            clave_cache = (nombre, round(sensibilidad, 2))

            if clave_cache in st.session_state["cache_analisis_vivo"]:
                img_bgr, res, latencia_ms = st.session_state["cache_analisis_vivo"][
                    clave_cache
                ]
            else:
                img_bgr = cv2.imread(ruta_img, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    continue
                t0 = time.perf_counter()
                res = analizar_imagen_real(img_bgr, sensibilidad=sensibilidad)
                latencia_ms = (time.perf_counter() - t0) * 1000.0
                st.session_state["cache_analisis_vivo"][clave_cache] = (
                    img_bgr,
                    res,
                    latencia_ms,
                )

            tiempo_total_ms += latencia_ms

            if not res["hay_tapon"]:
                estado = "SIN TAPÓN"
            elif res["pct_zona_critica"] >= umbral_alerta:
                estado = "ALERTA"
            elif res["pct_zona_critica"] >= umbral_precaucion:
                estado = "PRECAUCIÓN"
            else:
                estado = "ESTABLE"

            resultados.append(
                {
                    "Frame": nombre,
                    "Tapón": "Sí" if res["hay_tapon"] else "No",
                    "Rugosidad": res["indice_rugosidad"],
                    "Área Crítica (%)": res["pct_zona_critica"],
                    "Latencia (ms)": round(latencia_ms, 1),
                    "Estado": estado,
                }
            )
            imagenes_procesadas[nombre] = (img_bgr, res, estado, latencia_ms)

        if resultados:
            df_seq = pd.DataFrame(resultados)

            racha = 0
            rachas = []
            frame_disparo_auto = None
            for _, row in df_seq.iterrows():
                if row["Estado"] == "ALERTA":
                    racha += 1
                else:
                    racha = 0
                rachas.append(racha)
                if racha >= frames_disparo_req and frame_disparo_auto is None:
                    frame_disparo_auto = row["Frame"]

            df_seq["Racha Alerta"] = rachas
            max_racha = max(rachas) if rachas else 0
            paro_recomendado = max_racha >= frames_disparo_req
            total_alertas = int((df_seq["Estado"] == "ALERTA").sum())
            total_precaucion = int((df_seq["Estado"] == "PRECAUCIÓN").sum())
            peor_fila = df_seq.loc[df_seq["Área Crítica (%)"].idxmax()]

            if paro_recomendado:
                st.error(
                    f"🛑 **INTERLOCK PLC: STOP [0] — PARO DE PRODUCCIÓN OBLIGATORIO** | "
                    f"Se confirmaron **{max_racha} frames consecutivos** en ALERTA (disparo en `{frame_disparo_auto}`). "
                    f"Daño máximo: **{peor_fila['Área Crítica (%)']:.1f}%** en `{peor_fila['Frame']}`."
                )
            elif total_alertas > 0:
                st.warning(
                    f"⚠️ **INTERLOCK PLC: RUN [1] (PRECAUCIÓN) — Alerta aislada filtrada** | "
                    f"Se detectó daño puntual ({total_alertas} frame(s)), pero no alcanza los **{frames_disparo_req} cuadros consecutivos** para detener la línea."
                )
            else:
                st.success(
                    f"✅ **INTERLOCK PLC: RUN [1] — TAPÓN APTO PARA PRODUCCIÓN** | "
                    f"Secuencia conforme en los {len(df_seq)} cuadros evaluados (máx. {peor_fila['Área Crítica (%)']:.1f}%)."
                )

            with st.container(border=True):
                st.markdown(
                    f"**Dictamen de Calidad para `{tapon_id}` — Guardar y pasar al siguiente tapón**"
                )
                col_ap, col_det = st.columns(2)

                with col_ap:
                    obs_aprobado = st.text_input(
                        "Observaciones de liberación (opcional)",
                        value=(
                            "BLOQUEADO — Defecto confirmado (enviar a Revisión o Chatarrería)"
                            if paro_recomendado
                            else "Superficie conforme dentro de tolerancia"
                        ),
                        disabled=paro_recomendado,
                        key=f"obs_aprobado_input_{paro_recomendado}",
                    )
                    if st.button(
                        "✅ APROBAR Y GUARDAR SECUENCIA",
                        type="primary" if not paro_recomendado else "secondary",
                        disabled=paro_recomendado,
                        use_container_width=True,
                    ):
                        ahora_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        lote_limpio = lote_id.strip() or "SIN-LOTE"
                        tapon_limpio = tapon_id.strip() or "SIN-ID"

                        registro_ok = {
                            "Fecha_Hora": ahora_str,
                            "Lote_Orden": lote_limpio,
                            "ID_Tapon": tapon_limpio,
                            "Turno": turno_sel,
                            "Dictamen": "Aprobado (Conforme)",
                            "Frames_Analizados": int(len(df_seq)),
                            "Frames_Precaucion": total_precaucion,
                            "Frames_Alerta_Aislada": total_alertas,
                            "Area_Critica_Max_Pct": round(
                                float(peor_fila["Área Crítica (%)"]), 2
                            ),
                            "Area_Critica_Prom_Pct": round(
                                float(df_seq["Área Crítica (%)"].mean()), 2
                            ),
                            "Rugosidad_Max": round(
                                float(df_seq["Rugosidad"].max()), 2
                            ),
                            "Observaciones": obs_aprobado,
                        }
                        guardar_fila_csv(RUTA_APROBADOS, registro_ok, COLS_APROBADOS)
                        anexar_frames_historial(
                            df_seq,
                            lote_limpio,
                            tapon_limpio,
                            turno_sel,
                            "APROBADO",
                            ahora_str,
                        )

                        st.session_state["ultimo_evento_info"] = {
                            "tipo": "APROBADO",
                            **registro_ok,
                        }
                        st.session_state["pending_tapon_id"] = incrementar_id_tapon(
                            tapon_limpio
                        )
                        limpiar_area_activa()
                        st.rerun()

                    if paro_recomendado:
                        st.caption(
                            "🔒 **Aprobación bloqueada por política de calidad:** Toda secuencia con falla confirmada debe enviarse directamente a Revisión de Calidad o Chatarrería."
                        )

                with col_det:
                    d_dest, d_mot = st.columns([1.1, 1.4])
                    with d_dest:
                        destino_disp = st.selectbox(
                            "Destino no conforme:",
                            [
                                "Revisión de Calidad (Cuarentena)",
                                "Chatarrería (Scrap)",
                            ],
                        )
                    with d_mot:
                        motivo_default = (
                            f"Paro automático: {max_racha} frames consecutivos en alerta (máx {peor_fila['Área Crítica (%)']:.1f}% en {peor_fila['Frame']})"
                            if paro_recomendado
                            else f"Paro preventivo por inspección visual (máx {peor_fila['Área Crítica (%)']:.1f}% en {peor_fila['Frame']})"
                        )
                        motivo_paro = st.text_input(
                            "Motivo de detención", value=motivo_default
                        )

                    if st.button(
                        "🛑 DETENER PROCESO Y ENVIAR",
                        type="primary" if paro_recomendado else "secondary",
                        use_container_width=True,
                    ):
                        ahora_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        lote_limpio = lote_id.strip() or "SIN-LOTE"
                        tapon_limpio = tapon_id.strip() or "SIN-ID"

                        registro_stop = {
                            "Fecha_Hora": ahora_str,
                            "Lote_Orden": lote_limpio,
                            "ID_Tapon": tapon_limpio,
                            "Turno": turno_sel,
                            "Disposicion": destino_disp,
                            "Tipo_Paro": (
                                "Automático (Interlock IA)"
                                if paro_recomendado
                                else "Manual (Operador)"
                            ),
                            "Frames_Analizados": int(len(df_seq)),
                            "Frames_Alerta": total_alertas,
                            "Max_Racha_Consecutiva": int(max_racha),
                            "Area_Critica_Max_Pct": round(
                                float(peor_fila["Área Crítica (%)"]), 2
                            ),
                            "Rugosidad_Max": round(
                                float(df_seq["Rugosidad"].max()), 2
                            ),
                            "Frame_Critico": peor_fila["Frame"],
                            "Motivo": motivo_paro,
                        }
                        guardar_fila_csv(
                            RUTA_CUARENTENA, registro_stop, COLS_CUARENTENA
                        )
                        anexar_frames_historial(
                            df_seq,
                            lote_limpio,
                            tapon_limpio,
                            turno_sel,
                            f"DETENIDO ({destino_disp})",
                            ahora_str,
                        )

                        st.session_state["ultimo_evento_info"] = {
                            "tipo": "DETENIDO",
                            **registro_stop,
                        }
                        st.session_state["pending_tapon_id"] = incrementar_id_tapon(
                            tapon_limpio
                        )
                        limpiar_area_activa()
                        st.rerun()

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Frames Evaluados", f"{len(df_seq)}")
            k2.metric(
                "Frames en Alerta",
                f"{total_alertas}",
                delta=f"Racha máx: {max_racha} seguidos",
                delta_color="inverse" if total_alertas > 0 else "normal",
            )
            k3.metric(
                "Frame Más Crítico",
                peor_fila["Frame"],
                delta=f"{peor_fila['Área Crítica (%)']:.1f}% área crítica",
                delta_color="inverse",
            )
            k4.metric(
                "Latencia Promedio",
                f"{(tiempo_total_ms / len(df_seq)):.1f} ms/frame",
            )

            st.subheader("Evolución de Área Crítica por Frame")
            fig_seq, ax_seq = plt.subplots(figsize=(10, 2.8))
            ax_seq.plot(
                df_seq["Frame"],
                df_seq["Área Crítica (%)"],
                marker="o",
                color="crimson",
                label="Área Crítica (%)",
            )
            ax_seq.axhline(
                umbral_alerta,
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"Umbral Paro ({umbral_alerta:.1f}%)",
            )
            ax_seq.axhline(
                umbral_precaucion,
                color="orange",
                linestyle=":",
                alpha=0.7,
                label=f"Precaución ({umbral_precaucion:.1f}%)",
            )
            ax_seq.set_ylabel("Área Crítica (%)")
            ax_seq.tick_params(axis="x", rotation=45, labelsize=8)
            ax_seq.legend(fontsize=8)
            ax_seq.grid(alpha=0.3)
            st.pyplot(fig_seq)
            plt.close(fig_seq)

            st.subheader(f"Tabla de Resultados — Secuencia Activa ({tapon_id})")
            st.dataframe(df_seq, use_container_width=True)

            st.divider()

            st.subheader("Inspección Visual Detallada por Frame")
            opciones_frames = df_seq["Frame"].tolist()
            frame_objetivo = (
                frame_disparo_auto
                if frame_disparo_auto is not None
                else peor_fila["Frame"]
            )
            clave_secuencia = (tuple(opciones_frames), frame_objetivo)

            if len(opciones_frames) == 1:
                frame_sel = opciones_frames[0]
                st.caption(f"Mostrando captura única: **{frame_sel}**")
            else:
                if st.session_state.get("clave_secuencia_vista") != clave_secuencia:
                    st.session_state["clave_secuencia_vista"] = clave_secuencia
                    st.session_state["slider_frame_planta"] = frame_objetivo

                frame_sel = st.select_slider(
                    "Desliza manualmente para auditar cualquier cuadro (posicionado inicialmente en el frame crítico):",
                    options=opciones_frames,
                    key="slider_frame_planta",
                )

            img_bgr, res_sel, estado_sel, lat_sel = imagenes_procesadas[frame_sel]

            if estado_sel == "SIN TAPÓN":
                st.info(
                    f"**{frame_sel} — SIN TAPÓN DETECTADO:** Cuadro vacío o fondo de línea "
                    f"(0 falsos positivos) | Tiempo: {lat_sel:.1f} ms"
                )
            elif estado_sel == "ALERTA":
                st.error(
                    f"**{frame_sel} — DEFECTO CRÍTICO DETECTADO:** "
                    f"{res_sel['pct_zona_critica']:.1f}% de área dañada | "
                    f"Rugosidad: {res_sel['indice_rugosidad']} | Tiempo: {lat_sel:.1f} ms"
                )
            elif estado_sel == "PRECAUCIÓN":
                st.warning(
                    f"**{frame_sel} — PRECAUCIÓN:** {res_sel['pct_zona_critica']:.1f}% área crítica | "
                    f"Rugosidad: {res_sel['indice_rugosidad']} | Tiempo: {lat_sel:.1f} ms"
                )
            else:
                st.success(
                    f"**{frame_sel} — SUPERFICIE ESTABLE:** {res_sel['pct_zona_critica']:.1f}% área crítica | "
                    f"Rugosidad: {res_sel['indice_rugosidad']} | Tiempo: {lat_sel:.1f} ms"
                )

            c1, c2 = st.columns(2)
            with c1:
                st.image(
                    img_bgr,
                    channels="BGR",
                    caption=f"1. Original ({frame_sel})",
                    use_container_width=True,
                )
                st.image(
                    res_sel["sin_gotas"],
                    caption="2. Filtrado (Bilateral anti-ruido + CLAHE)",
                    use_container_width=True,
                )
            with c2:
                st.image(
                    res_sel["heatmap_bgr"],
                    channels="BGR",
                    caption="3. Mapa de Relieve (Azul=Liso | Amarillo/Rojo=Rugosidad)",
                    use_container_width=True,
                )
                st.image(
                    res_sel["superpuesta_bgr"],
                    channels="BGR",
                    caption="4. Vista Asistida para el Operador (Silueta YOLOv8-seg)",
                    use_container_width=True,
                )

    st.divider()

    st.subheader("Base de Datos Acumulada de Producción y Calidad")
    df_aprobados = cargar_csv_seguro(RUTA_APROBADOS, COLS_APROBADOS)
    df_cuarentena = cargar_csv_seguro(RUTA_CUARENTENA, COLS_CUARENTENA)
    df_hist_frames = (
        pd.read_csv(RUTA_HISTORIAL_FRAMES)
        if os.path.exists(RUTA_HISTORIAL_FRAMES)
        else pd.DataFrame()
    )

    total_aprobados = len(df_aprobados)
    total_detenidas = len(df_cuarentena)
    total_procesados = total_aprobados + total_detenidas
    yield_pct = (
        (total_aprobados / total_procesados) * 100.0 if total_procesados > 0 else 100.0
    )
    en_revision = (
        int(df_cuarentena["Disposicion"].str.contains("Revisión", na=False).sum())
        if not df_cuarentena.empty
        else 0
    )
    en_scrap = (
        int(df_cuarentena["Disposicion"].str.contains("Chatarrería", na=False).sum())
        if not df_cuarentena.empty
        else 0
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Tapones Procesados", f"{total_procesados}")
    m2.metric("Aprobados (Conformes)", f"{total_aprobados}")
    m3.metric("Tasa de Aprobación (Yield)", f"{yield_pct:.1f}%")
    m4.metric("En Revisión de Calidad", f"{en_revision}")
    m5.metric("En Chatarrería (Scrap)", f"{en_scrap}")

    subtab_ok, subtab_stop = st.tabs(
        [
            f"✅ Tapones Aprobados ({total_aprobados})",
            f"🛑 Revisión de Calidad y Chatarrería ({total_detenidas})",
        ]
    )

    with subtab_ok:
        if df_aprobados.empty:
            st.caption(
                "Aún no hay secuencias aprobadas guardadas. Evalúa un tapón conforme arriba y presiona '✅ APROBAR Y GUARDAR SECUENCIA'."
            )
        else:
            st.dataframe(df_aprobados, use_container_width=True)

    with subtab_stop:
        if df_cuarentena.empty:
            st.caption("Aún no hay secuencias enviadas a Revisión o Chatarrería.")
        else:
            st.dataframe(df_cuarentena, use_container_width=True)

    resumen_meta = {
        "Fecha de Generación": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Lote / Orden Activa": lote_id,
        "Siguiente ID Tapón": tapon_id,
        "Turno": turno_sel,
        "Total Tapones Procesados": total_procesados,
        "Tapones Aprobados": total_aprobados,
        "Tasa de Aprobación / Yield (%)": round(yield_pct, 2),
        "En Revisión de Calidad": en_revision,
        "En Chatarrería (Scrap)": en_scrap,
        "Umbral Defecto (%)": umbral_alerta,
        "Frames Consecutivos Disparo": frames_disparo_req,
        "Sensibilidad Relieve": sensibilidad,
    }

    bytes_excel = generar_reporte_excel(
        df_aprobados=df_aprobados,
        df_cuarentena=df_cuarentena,
        df_hist_frames=df_hist_frames,
        df_seq_activa=df_seq,
        resumen_meta=resumen_meta,
    )

    exp1, exp2 = st.columns([2, 1])
    with exp1:
        if bytes_excel is not None:
            st.download_button(
                label="📊 Descargar Base de Datos Completa en Excel (.xlsx)",
                data=bytes_excel,
                file_name=f"Auditoria_Produccion_Tapones_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                disabled=(total_procesados == 0 and df_seq is None),
            )
        else:
            st.warning(
                "Para descargar en Excel (.xlsx), ejecuta `pip install openpyxl` en tu terminal."
            )
    with exp2:
        if total_procesados > 0:
            if st.button(
                "Reiniciar bases de datos (Nueva demo)", use_container_width=True
            ):
                for ruta_db in (
                    RUTA_APROBADOS,
                    RUTA_CUARENTENA,
                    RUTA_HISTORIAL_FRAMES,
                ):
                    if os.path.exists(ruta_db):
                        os.remove(ruta_db)
                st.session_state["ultimo_evento_info"] = None
                st.session_state["pending_tapon_id"] = "TP-001"
                st.rerun()


# ==============================================================================
# PESTAÑA 3: STREAM EN VIVO DE ALTA VELOCIDAD (SIMULADOR HMI)
# ==============================================================================
with tab3:
    if "cache_stream_hmi" not in st.session_state:
        st.session_state["cache_stream_hmi"] = {}
    if "stream_uploader_key" not in st.session_state:
        st.session_state["stream_uploader_key"] = 0
    if "stream_activo" not in st.session_state:
        st.session_state["stream_activo"] = False

    st.subheader("Simulador de Flujo en Tiempo Real (Monitor HMI de Planta)")
    st.write(
        "Simula el flujo continuo de cámara industrial con segmentación YOLOv8-seg, "
        "mapa de relieve y gráfica telemétrica de baja latencia (< 2 ms/frame)."
    )

    up_s1, up_s2 = st.columns([4.5, 1.1])
    with up_s1:
        archivos_stream = st.file_uploader(
            "Cargar secuencia de frames para el stream (.jpg, .png)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key=f"uploader_stream_hmi_{st.session_state['stream_uploader_key']}",
        )
    with up_s2:
        st.write("")
        st.write("")
        if st.button(
            "🗑️ Limpiar",
            key="btn_limpiar_stream",
            use_container_width=True,
            help="Borrar la secuencia cargada y limpiar la caché del simulador",
        ):
            st.session_state["stream_activo"] = False
            st.session_state["cache_stream_hmi"].clear()
            st.session_state["stream_uploader_key"] += 1
            st.rerun()

    if not archivos_stream:
        st.session_state["stream_activo"] = False

    sc1, sc2, sc3, sc4 = st.columns([1.4, 1.4, 1.4, 1.4])
    fps_objetivo = sc1.slider(
        "Velocidad de Stream (FPS)",
        min_value=1.0,
        max_value=15.0,
        value=5.0,
        step=0.5,
        help="Cuadros por segundo objetivo durante la reproducción.",
    )
    sensibilidad_s = sc2.slider(
        "Sensibilidad de relieve (HMI)",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1,
    )
    umbral_alerta_s = sc3.slider(
        "Umbral de Paro — Área Crítica (%)",
        min_value=1.0,
        max_value=30.0,
        value=8.0,
        step=0.5,
    )
    frames_disparo_s = sc4.slider(
        "Frames seguidos para STOP",
        min_value=1,
        max_value=5,
        value=2,
    )
    umbral_prec_s = umbral_alerta_s * 0.5

    op1, op2, op3, op4, op5 = st.columns([1.3, 1.0, 1.4, 1.6, 1.6])
    with op1:
        if st.button(
            "▶ Iniciar Stream",
            type="primary",
            use_container_width=True,
            disabled=(not archivos_stream),
        ):
            st.session_state["stream_activo"] = True
    with op2:
        if st.button(
            "⏹ Detener",
            use_container_width=True,
            disabled=(not archivos_stream),
        ):
            st.session_state["stream_activo"] = False
    with op3:
        reproducir_bucle = st.checkbox(
            "🔁 Proyectar en bucle",
            value=True,
            help="Una vez procesada la secuencia, la proyecta continuamente desde el cuadro 1 hasta presionar Detener.",
        )
    with op4:
        auto_stop_interlock = st.checkbox(
            "Congelar al disparar STOP",
            value=True,
            help="Detiene el bucle en el cuadro exacto donde se confirma el defecto. Desmárcalo para proyección continua sin pausas.",
        )
    with op5:
        preprocesar_antes = st.checkbox(
            "Pre-cargar IA (FPS estables)",
            value=True,
            help="Analiza los cuadros antes de iniciar para proyectar el bucle a velocidad constante sin tirones.",
        )

    visor_hmi = st.empty()

    def obtener_frame_procesado(arch_obj, sens: float):
        clave_s = (arch_obj.name, arch_obj.size, round(sens, 2))
        if clave_s in st.session_state["cache_stream_hmi"]:
            return st.session_state["cache_stream_hmi"][clave_s]

        bytes_arr = np.frombuffer(arch_obj.getvalue(), dtype=np.uint8)
        img_bgr = cv2.imdecode(bytes_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None

        t_inf0 = time.perf_counter()
        res_s = analizar_imagen_real(img_bgr, sensibilidad=sens)
        lat_inf_ms = (time.perf_counter() - t_inf0) * 1000.0

        paquete = (
            res_s["hay_tapon"],
            float(res_s["pct_zona_critica"]),
            float(res_s["indice_rugosidad"]),
            lat_inf_ms,
            redimensionar_preview(img_bgr, 480),
            redimensionar_preview(res_s["superpuesta_bgr"], 480),
            redimensionar_preview(res_s["heatmap_bgr"], 480),
        )
        st.session_state["cache_stream_hmi"][clave_s] = paquete
        return paquete

    if not archivos_stream:
        visor_hmi.info(
            "Sube una secuencia de imágenes arriba (por ejemplo, `Frame_01.jpg` a `Frame_35.jpg`) "
            "y presiona **▶ Iniciar Stream** para simular el monitoreo continuo."
        )
    elif st.session_state["stream_activo"]:

        def orden_natural_archivo(arch_obj):
            nums = re.findall(r"\d+", arch_obj.name)
            return int(nums[0]) if nums else arch_obj.name

        secuencia_ordenada = sorted(archivos_stream, key=orden_natural_archivo)
        total_frames_s = len(secuencia_ordenada)
        intervalo_objetivo = 1.0 / fps_objetivo

        if preprocesar_antes:
            pendientes = [
                a
                for a in secuencia_ordenada
                if (a.name, a.size, round(sensibilidad_s, 2))
                not in st.session_state["cache_stream_hmi"]
            ]
            if pendientes:
                with visor_hmi.container():
                    barra_prep = st.progress(
                        0.0,
                        text=f"Procesando visión IA para proyección continua (0/{len(pendientes)} cuadros nuevos)...",
                    )
                    for i_p, arch_p in enumerate(pendientes, start=1):
                        obtener_frame_procesado(arch_p, sensibilidad_s)
                        barra_prep.progress(
                            i_p / len(pendientes),
                            text=f"Procesando visión IA para proyección continua ({i_p}/{len(pendientes)} cuadros nuevos)...",
                        )

        ciclo_bucle = 1
        while st.session_state["stream_activo"]:
            historial_frames_nombres = []
            historial_area_critica = []
            racha_s = 0
            interlock_activado = False
            frame_corte = None

            for idx, arch in enumerate(secuencia_ordenada, start=1):
                t_frame_inicio = time.perf_counter()
                paquete = obtener_frame_procesado(arch, sensibilidad_s)
                if paquete is None:
                    continue

                (
                    hay_tapon,
                    area_actual,
                    _,
                    lat_inf_ms,
                    prev_orig,
                    prev_seg,
                    prev_heat,
                ) = paquete

                if not hay_tapon:
                    estado_f = "SIN TAPÓN"
                    racha_s = 0
                elif area_actual >= umbral_alerta_s:
                    estado_f = "ALERTA"
                    racha_s += 1
                elif area_actual >= umbral_prec_s:
                    estado_f = "PRECAUCIÓN"
                    racha_s = 0
                else:
                    estado_f = "ESTABLE"
                    racha_s = 0

                historial_frames_nombres.append(arch.name)
                historial_area_critica.append(area_actual)

                if racha_s >= frames_disparo_s:
                    interlock_activado = True
                    frame_corte = arch.name

                grafica_bgr = dibujar_grafica_hmi(
                    historial_frames_nombres,
                    historial_area_critica,
                    umbral_alerta_s,
                    umbral_prec_s,
                    total_frames_s,
                )

                tiempo_procesado = time.perf_counter() - t_frame_inicio
                fps_real = 1.0 / max(tiempo_procesado, intervalo_objetivo)
                etiqueta_bucle = f" | Bucle #{ciclo_bucle}" if reproducir_bucle else ""

                with visor_hmi.container():
                    st.progress(
                        idx / total_frames_s,
                        text=f"Proyección en vivo: cuadro {idx} de {total_frames_s} ({arch.name}){etiqueta_bucle}",
                    )

                    if interlock_activado:
                        st.error(
                            f"🛑 **INTERLOCK PLC: STOP [0] — DEFECTO CONFIRMADO EN `{arch.name}`** | "
                            f"Se detectaron **{racha_s} cuadros consecutivos** sobre el umbral ({area_actual:.1f}% área crítica)."
                        )
                    elif estado_f == "ALERTA":
                        st.warning(
                            f"⚠️ **ALERTA PUNTUAL ({racha_s}/{frames_disparo_s}) — PLC: RUN [1]** | "
                            f"`{arch.name}` registró {area_actual:.1f}% área crítica (evaluando confirmación en siguiente cuadro)."
                        )
                    elif estado_f == "PRECAUCIÓN":
                        st.warning(
                            f"⚠️ **PRECAUCIÓN — PLC: RUN [1]** | `{arch.name}` con {area_actual:.1f}% área crítica."
                        )
                    elif estado_f == "SIN TAPÓN":
                        st.info(
                            f"ℹ️ **SIN TAPÓN EN CUADRO — PLC: RUN [1]** | `{arch.name}` (fondo de línea filtrado sin falsos positivos)."
                        )
                    else:
                        st.success(
                            f"✅ **OPERACIÓN NOMINAL — PLC: RUN [1]** | `{arch.name}` estable ({area_actual:.1f}% área crítica)."
                        )

                    m_s1, m_s2, m_s3, m_s4 = st.columns(4)
                    m_s1.metric("Cuadro en Vivo", f"{arch.name} ({idx}/{total_frames_s})")
                    m_s2.metric(
                        "Velocidad Stream",
                        f"{fps_real:.1f} FPS",
                        delta=f"Inferencia IA: {lat_inf_ms:.0f} ms",
                    )
                    m_s3.metric(
                        "Área Crítica Actual",
                        f"{area_actual:.1f}%",
                        delta=f"Umbral: {umbral_alerta_s:.1f}%",
                        delta_color="inverse" if area_actual >= umbral_alerta_s else "normal",
                    )
                    m_s4.metric(
                        "Racha de Alerta",
                        f"{racha_s} / {frames_disparo_s}",
                    )

                    v_col1, v_col2, v_col3 = st.columns(3)
                    with v_col1:
                        st.image(
                            prev_orig,
                            channels="BGR",
                            caption=f"1. Cámara en Línea ({arch.name})",
                            use_container_width=True,
                        )
                    with v_col2:
                        st.image(
                            prev_seg,
                            channels="BGR",
                            caption="2. Detección y Silueta YOLOv8-seg",
                            use_container_width=True,
                        )
                    with v_col3:
                        st.image(
                            prev_heat,
                            channels="BGR",
                            caption="3. Mapa de Relieve en Tiempo Real",
                            use_container_width=True,
                        )

                    st.image(
                        grafica_bgr,
                        channels="BGR",
                        use_container_width=True,
                    )

                if interlock_activado and auto_stop_interlock:
                    st.session_state["stream_activo"] = False
                    st.toast(
                        f"🛑 Interlock activado en {frame_corte}: stream detenido automáticamente."
                    )
                    break

                tiempo_espera = max(
                    0.0, intervalo_objetivo - (time.perf_counter() - t_frame_inicio)
                )
                if tiempo_espera > 0:
                    time.sleep(tiempo_espera)

            if not st.session_state["stream_activo"]:
                break

            if not reproducir_bucle:
                st.session_state["stream_activo"] = False
                if not interlock_activado:
                    st.toast("✅ Secuencia completada.")
                break

            ciclo_bucle += 1
    else:
        visor_hmi.info(
            f"**{len(archivos_stream)} cuadro(s) listo(s)** en cola. Ajusta los FPS arriba y haz clic en **▶ Iniciar Stream**."
        )