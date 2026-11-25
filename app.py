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
os.makedirs(CARPETA_VIVO, exist_ok=True)

st.title("Sistema de Visión y Analítica Predictiva — Tapón Perforador")
st.caption("Enfoque Entradas → Transformación → Salidas | Ciclo PDCA & Human-in-the-Loop")


@st.cache_data
def cargar_datos():
    return pd.read_csv("data/dataset_completo.csv")


@st.cache_resource
def cargar_modelos():
    clf = joblib.load("data/modelo_riesgo.joblib")
    reg = joblib.load("data/modelo_rul.joblib")
    return clf, reg


df = cargar_datos()
modelo_riesgo, modelo_rul = cargar_modelos()

tab1, tab2 = st.tabs(
    ["Simulación de Corrida (Predictivo)", "Inspección de Imagen Real (Edge)"]
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

    # Medición de tiempo real de procesamiento (OpenCV + LightGBM)
    t_inicio = time.perf_counter()
    img_cruda = generar_cuadro_tapon(
        desgaste=float(fila["desgaste_real"]), seed=idx_global
    )
    vis = analizar_cuadro(img_cruda)

    X_actual = pd.DataFrame([fila[FEATURES]])
    prob_riesgo = float(modelo_riesgo.predict_proba(X_actual)[0, 1]) * 100.0
    rul_estimado = max(0.0, float(modelo_rul.predict(X_actual)[0]))
    tiempo_ms = (time.perf_counter() - t_inicio) * 1000.0

    # KPIs superiores
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Riesgo de Defecto", f"{prob_riesgo:.1f}%")
    col2.metric("Vida Útil Restante (RUL)", f"{rul_estimado:.1f} ciclos")
    col3.metric("Temperatura", f"{fila['temperatura_c']:.1f} °C")
    col4.metric("Carga de Motor", f"{fila['carga_motor_pct']:.1f} %")
    col5.metric("Latencia Edge", f"{tiempo_ms:.1f} ms", delta="Ventana: 5000 ms")

    # Alerta explicable
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

    # Diagnóstico visual
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

    # Tendencias por corrida
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

    # Registro Human-in-the-Loop
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
# PESTAÑA 2: INSPECCIÓN DE IMAGEN REAL (CÁMARA REMOTA + YOLOv8-SEG EN VIVO)
# ==============================================================================
with tab2:
    if "archivos_guardados_ids" not in st.session_state:
        st.session_state["archivos_guardados_ids"] = set()
    if "cache_analisis_vivo" not in st.session_state:
        st.session_state["cache_analisis_vivo"] = {}
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0

    st.subheader("Inspección en Tiempo Real (Celular / Cámara → Monitor de Planta)")
    st.write(
        "Desde tu celular o desde esta PC, toma una foto o selecciona varias imágenes a la vez. "
        "Todas se sincronizan automáticamente en la secuencia en vivo."
    )

    col_up, col_btn = st.columns([4, 1])
    with col_up:
        archivos = st.file_uploader(
            "Tomar foto con cámara o seleccionar secuencia (.jpg, .png)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key=f"uploader_vivo_{st.session_state['uploader_key']}",
        )
    with col_btn:
        st.write("")
        st.write("")
        if st.button("Limpiar secuencia", use_container_width=True):
            for nombre in os.listdir(CARPETA_VIVO):
                ruta_f = os.path.join(CARPETA_VIVO, nombre)
                if os.path.isfile(ruta_f):
                    os.remove(ruta_f)
            st.session_state["archivos_guardados_ids"].clear()
            st.session_state["cache_analisis_vivo"].clear()
            st.session_state["uploader_key"] += 1
            st.rerun()

    # Guardar archivos recién subidos en la carpeta compartida (sin duplicar en reruns)
    if archivos:
        nuevos_guardados = 0
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
            nuevos_guardados += 1

        if nuevos_guardados > 0:
            st.toast(f"{nuevos_guardados} imagen(es) enviada(s) al monitor en vivo.")

    cal1, cal2, cal3 = st.columns([2, 2, 1.5])
    sensibilidad = cal1.slider(
        "Sensibilidad de relieve", min_value=0.5, max_value=2.0, value=1.0, step=0.1
    )
    umbral_alerta = cal2.slider(
        "Umbral de Alerta — Área Crítica (%)",
        min_value=1.0,
        max_value=30.0,
        value=8.0,
        step=0.5,
    )
    umbral_precaucion = umbral_alerta * 0.5
    auto_sync = cal3.checkbox(
        "Auto-actualizar monitor (2 s)",
        value=True,
        help="Refresca automáticamente esta pantalla cuando llegan fotos desde el celular.",
    )

    def renderizar_monitor_vivo():
        archivos_disco = [
            f
            for f in os.listdir(CARPETA_VIVO)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not archivos_disco:
            st.info(
                "Esperando imágenes... Toma una foto desde tu celular o sube archivos arriba para iniciar la inspección."
            )
            return

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

        if not resultados:
            return

        df_seq = pd.DataFrame(resultados)
        peor_fila = df_seq.loc[df_seq["Área Crítica (%)"].idxmax()]
        ultima_fila = df_seq.iloc[-1]

        # KPIs globales de la secuencia
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Capturas en Secuencia", f"{len(df_seq)}")
        k2.metric(
            "Última Captura",
            ultima_fila["Frame"],
            delta=f"{ultima_fila['Estado']} ({ultima_fila['Área Crítica (%)']:.1f}%)",
            delta_color="inverse" if ultima_fila["Estado"] == "ALERTA" else "normal",
        )
        k3.metric(
            "Frame Más Crítico",
            peor_fila["Frame"],
            delta=f"{peor_fila['Área Crítica (%)']:.1f}% área crítica",
            delta_color="inverse",
        )
        k4.metric("Tiempo Acumulado", f"{tiempo_total_ms:.1f} ms")

        # Gráfica de evolución a lo largo de la secuencia
        st.subheader("Evolución de Rugosidad y Área Crítica por Captura")
        fig_seq, ax_seq = plt.subplots(figsize=(10, 3))
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
            alpha=0.6,
            label=f"Umbral Alerta ({umbral_alerta:.1f}%)",
        )
        ax_seq.axhline(
            umbral_precaucion,
            color="orange",
            linestyle="--",
            alpha=0.6,
            label=f"Umbral Precaución ({umbral_precaucion:.1f}%)",
        )
        ax_seq.set_ylabel("Área Crítica (%)")
        ax_seq.tick_params(axis="x", rotation=45, labelsize=8)
        ax_seq.legend(fontsize=8)
        ax_seq.grid(alpha=0.3)
        st.pyplot(fig_seq)
        plt.close(fig_seq)

        # Tabla con resultados individuales de cada frame
        st.subheader("Historial de Capturas Analizadas")
        st.dataframe(df_seq, use_container_width=True)

        st.divider()

        # Inspector visual (muestra por defecto la captura más reciente)
        st.subheader("Inspección Visual Detallada")
        opciones_frames = df_seq["Frame"].tolist()
        if len(opciones_frames) == 1:
            frame_sel = opciones_frames[0]
            st.caption(f"Mostrando captura actual: **{frame_sel}**")
        else:
            frame_sel = st.select_slider(
                "Desliza para revisar cualquier captura (por defecto muestra la más reciente):",
                options=opciones_frames,
                value=opciones_frames[-1],
            )

        img_bgr, res_sel, estado_sel, lat_sel = imagenes_procesadas[frame_sel]

        if estado_sel == "SIN TAPÓN":
            st.info(
                f"**{frame_sel} — SIN TAPÓN DETECTADO:** Cuadro vacío o fondo de línea "
                f"(0 falsos positivos) | Tiempo: {lat_sel:.1f} ms"
            )
        elif estado_sel == "ALERTA":
            st.error(
                f"**{frame_sel} — TAPÓN DEFECTUOSO (DETENER / CAMBIAR):** "
                f"{res_sel['pct_zona_critica']:.1f}% de superficie con daño crítico | "
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

    @st.fragment(run_every=2)
    def monitor_auto():
        renderizar_monitor_vivo()

    if auto_sync:
        monitor_auto()
    else:
        renderizar_monitor_vivo()