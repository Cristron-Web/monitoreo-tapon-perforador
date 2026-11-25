# Sistema de Visión y Analítica Predictiva — Tapón Perforador

Aplicación de **Visión Artificial (`YOLOv8-seg` + OpenCV)** y **Mantenimiento Predictivo** para la inspección superficial y estimación de desgaste en tapones perforadores.

## Demo Interactiva en Vivo

👉 **[Abrir Aplicación Web (Streamlit Cloud)](https://monitoreo-tapon-perforador-krsrq26hj5c2uedpbr9kle.streamlit.app/)**

---

### Indicaciones de acceso y tiempos de espera

Debido a que la aplicación está alojada en un servidor gratuito en la nube, ten en cuenta lo siguiente al abrir el enlace:

1. **Si la aplicación aparece suspendida (*"This app has gone to sleep"*):**
   * Haz clic en el botón azul **"Yes, get this app back up!"**.
   * Espera entre **1 y 2 minutos** mientras el servidor reactiva el entorno e inicializa las librerías de visión artificial (`PyTorch` y `Ultralytics`). La página recargará automáticamente al terminar.
2. **Primera inferencia en inspección real:**
   * Al subir la primera imagen en la pestaña de inspección, el análisis puede tardar **2 a 4 segundos** mientras carga en memoria los pesos del modelo (`models/best.pt`). Los cuadros siguientes se procesan de forma mucho más rápida.

---

### Guía rápida de evaluación

Dentro de la aplicación encontrarás dos pestañas principales:

* **1. Simulación de Corrida (Predictivo):** Desliza el control lateral de *Ciclo de operación* para evaluar en tiempo real el **Riesgo de Defecto (%)**, la **Vida Útil Restante (RUL)**, el ajuste geométrico de elipse y los mapas topográficos (*Hillshade*).
* **2. Inspección de Imagen Real (Edge):** Carga una o varias fotografías de tapones perforadores (`.jpg` o `.png`) para visualizar la segmentación automática con **YOLOv8-seg** (contorno cian), el filtrado de ruido y el mapa de calor de zonas críticas.
