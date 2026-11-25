# Manual de Usuario

## Sistema de Vision y Analitica Predictiva — Tapón Perforador

**Version del manual:** 1.0  
**Fecha:** 25 de noviembre de 2026

---

## 1. Objetivo del sistema

Este proyecto ayuda a supervisar el estado de un tapón perforador mediante dos modalidades:

1. **Simulación de corrida predictiva:** analiza datos históricos o simulados de cada corrida y estima el riesgo de defecto y la vida útil restante.
2. **Inspección de imagen real:** procesa fotografías del tapón con segmentación YOLOv8-seg y OpenCV para detectar daño superficial y abolladuras.

El sistema funciona como apoyo a la decisión. Los resultados deben ser validados por personal operativo y de mantenimiento conforme a los procedimientos seguros de la planta.

## 2. Perfil de usuario

| Perfil | Responsabilidades principales |
|---|---|
| Operador de línea | Revisar indicadores, cargar imágenes y registrar decisiones |
| Supervisor | Interpretar alertas, definir acciones y revisar el historial PDCA |
| Mantenimiento | Confirmar inspecciones, programar cambios y documentar hallazgos |
| Administrador técnico | Mantener los modelos, datos, entorno y archivos de captura |

## 3. Requisitos

### Requisitos recomendados

- Windows con Python 3.13.
- Navegador web moderno.
- Memoria RAM de al menos 8 GB.
- Acceso a la carpeta del proyecto.
- Conexión en red local si se desea cargar imágenes desde un celular.

### Componentes incluidos

- `app.py`: aplicación web en Streamlit.
- `src/simulador.py`: genera telemetría e imágenes simuladas.
- `src/vision.py`: análisis con OpenCV y YOLOv8-seg.
- `src/modelo.py`: construye el conjunto de datos y reentrena modelos LightGBM.
- `data/modelo_riesgo.joblib`: modelo de probabilidad de defecto.
- `data/modelo_rul.joblib`: modelo de vida útil restante.
- `data/dataset_completo.csv`: datos de corridas y características de visión.
- `models/best.pt`: modelo YOLOv8-seg de segmentación del tapón.
- `venv/`: entorno Python ya preparado.

## 4. Instalación y puesta en marcha

### 4.1. Abrir una terminal

Desde Visual Studio Code:

1. Abra la carpeta **Hackatec Nacional**.
2. Seleccione **Terminal > Nueva terminal**.
3. Confirme que la terminal se encuentre en la carpeta del proyecto.

### 4.2. Activar el entorno

En PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Si PowerShell bloquea la ejecución, use el símbolo del sistema:

```bat
venv\Scripts\activate.bat
```

### 4.3. Ejecutar la aplicación

```powershell
python -m streamlit run app.py
```

Abra en el navegador la dirección que muestre Streamlit, normalmente:

```text
http://localhost:8501
```

Para detener la aplicación, vuelva a la terminal y presione `Ctrl+C`.

## 5. Acceso desde un celular

Streamlit puede mostrar un enlace de red local al iniciar. En el terminal puede aparecer una URL similar a:

```text
Network URL: http://<IP-DE-LA-PC>:8501
```

1. Conecte el celular y la PC a la misma red.
2. Escriba la URL de red en el navegador del celular.
3. Permita el acceso a la cámara cuando el navegador lo solicite.
4. Use la pestaña **Inspección de Imagen Real**.

> Las imágenes se guardan en `data/capturas_vivo` dentro del equipo que ejecuta la aplicación.

## 6. Pantalla principal

La interfaz contiene dos pestañas:

- **Simulación de Corrida (Predictivo)**.
- **Inspección de Imagen Real (Edge)**.

La barra lateral controla la corrida seleccionada y el ciclo actual en el modo predictivo.

---

# 7. Simulación de Corrida Predictiva

## 7.1. Seleccionar una corrida

En la barra lateral:

1. Abra **Seleccionar Corrida ID**.
2. Elija una corrida disponible.
3. Los gráficos y los resultados se actualizarán para esa corrida.

## 7.2. Seleccionar el ciclo

Use el control **Ciclo de operación** para elegir la pieza o ciclo a revisar.

Al cambiar el ciclo se recalculan:

- Riesgo de defecto.
- Vida útil restante.
- Temperatura.
- Carga del motor.
- Diagnóstico visual.
- Tendencias de ovalamiento y rugosidad.

## 7.3. Interpretar los indicadores superiores

| Indicador | Significado |
|---|---|
| Riesgo de Defecto | Probabilidad estimada de que el tapón tenga defecto |
| Vida Útil Restante (RUL) | Ciclos estimados antes del límite operativo |
| Temperatura | Temperatura registrada en el ciclo seleccionado |
| Carga de Motor | Porcentaje de carga registrada |
| Latencia Edge | Tiempo de cálculo para la imagen y los modelos |

### Estados del riesgo predictivo

| Estado | Condición | Acción sugerida |
|---|---|---|
| Operación normal | Riesgo menor a 35% y RUL mayor a 5 ciclos | Continuar monitorizando |
| Precaución | Riesgo entre 35% y 69%, o condición progresiva | Inspección preventiva |
| Alerta crítica | Riesgo mayor o igual a 70%, o RUL menor o igual a 5 ciclos | Programar cambio del tapón |

Las alertas muestran los factores más relevantes: ovalamiento, rugosidad, temperatura o excentricidad.

## 7.4. Revisar el diagnóstico visual

La sección presenta cuatro imágenes:

1. **Entrada cruda:** imagen simulada a baja resolución y con ruido.
2. **Acondicionada:** resultado de CLAHE y suavizado gaussiano.
3. **Contorno:** segmentación Otsu, elipse ajustada y referencia ideal.
4. **Mapa de relieve:** representación topográfica con isolíneas.

Revise los cambios junto con los indicadores operativos; una geometría o una textura fuera de rango puede explicar el riesgo estimado.

## 7.5. Consultar tendencias

Los gráficos muestran:

- Ovalamiento real y promedio móvil de tres ciclos.
- Rugosidad calculada por Sobel y promedio móvil de tres ciclos.
- Línea roja que identifica el ciclo actual.

Una tendencia ascendente sostenida puede indicar desgaste progresivo, aunque el valor absoluto sea todavía bajo.

## 7.6. Registrar una decisión PDCA

1. Seleccione **Acción tomada**:
   - Continuar corrida.
   - Inspección visual preventiva.
   - Cambio de tapón programado.
   - Falsa alarma (recalibrar).
2. Escriba observaciones si son necesarias.
3. Presione **Guardar en historial PDCA**.

El registro se añade a:

```text
data/historial_operador.csv
```

Los cinco registros más recientes aparecen debajo del formulario.

Cada entrada incluye:

- Fecha y hora.
- ID de corrida.
- Ciclo.
- Riesgo estimado.
- RUL estimado.
- Decisión.
- Nota del operador.

---

# 8. Inspección de Imagen Real

## 8.1. Capturar o cargar imágenes

En **Inspección de Imagen Real**:

1. Presione **Tomar foto con cámara o seleccionar secuencia**.
2. Use la cámara del dispositivo o elija una o varias imágenes.
3. Los formatos admitidos son JPG, JPEG y PNG.
4. Las imágenes se copiarán automáticamente a `data/capturas_vivo`.

Para incluir varias capturas consecutivas, seleccione varios archivos en el mismo diálogo. En iOS o Android, la opción disponible depende del navegador.

## 8.2. Recomendaciones fotográficas

- Mantenga el tapón dentro del encuadre.
- Evite movimiento y desenfoque.
- Use iluminación uniforme.
- Evite reflejos intensos y sombras sobre el tapón.
- Mantenga una distancia estable entre capturas.
- Capture el cuerpo completo y la zona que se desea inspeccionar.
- Evita utilizar el flash si produce reflejos sobre la superficie.

## 8.3. Ajustar parámetros

### Sensibilidad de relieve

Controla el peso del mapa de fallas combinado. El valor inicial es `1.0`.

- `0.5`: menos sensible; reduce detalles débiles y posible sobrealerta.
- `1.0`: configuración inicial recomendada.
- `2.0`: más sensible; puede aumentar detecciones y falsos positivos.

### Umbral de Alerta — Área Crítica

Define el porcentaje mínimo de superficie crítica para generar una alerta. El valor inicial es `8.0%`.

El nivel de precaución se calcula automáticamente como la mitad del umbral de alerta.

Ejemplo con el valor inicial:

- Área crítica menor a 4.0%: ESTABLE.
- Área crítica de 4.0% a menos de 8.0%: PRECAUCIÓN.
- Área crítica mayor o igual a 8.0%: ALERTA.

## 8.4. Autoactualización

La opción **Auto-actualizar monitor (2 s)** está activada por defecto.

- **Activada:** actualiza el análisis cada dos segundos.
- **Desactivada:** actualiza el monitor únicamente al ejecutar una acción o recargar la página.

Use esta opción para vigilar imágenes recibidas desde celulares sin refrescar manualmente.

## 8.5. Limpiar la secuencia

Presione **Limpiar secuencia** para eliminar todas las imágenes JPEG y PNG de `data/capturas_vivo`.

Esta acción elimina las capturas guardadas en disco y reinicia la memoria de análisis de la sesión. No elimina:

- Modelos.
- Datos históricos.
- Historial de decisiones.

## 8.6. Interpretar la vista en vivo

### KPIs de secuencia

- **Capturas en Secuencia:** número de imágenes procesadas.
- **Última Captura:** nombre y estado de la imagen más reciente.
- **Frame Más Crítico:** captura con mayor área crítica.
- **Tiempo Acumulado:** suma de tiempos de análisis de las imágenes.

### Estado por captura

| Estado | Significado | Acción sugerida |
|---|---|---|
| ESTABLE | La zona crítica permanece por debajo del nivel de precaución | Continuar supervisión |
| PRECAUCIÓN | La zona crítica supera la mitad del umbral de alerta | Repetir captura e inspeccionar |
| ALERTA | La zona crítica alcanza el umbral configurado | Detener o cambiar según procedimiento |
| SIN TAPÓN | No se detectó una silueta válida | Repetir captura |

## 8.7. Consultar el historial de capturas

La tabla contiene:

- Nombre del frame.
- Si se detectó el tapón.
- Índice de rugosidad.
- Porcentaje de área crítica.
- Latencia.
- Estado.

El gráfico muestra la evolución del área crítica y las líneas de precaución y alerta.

## 8.8. Revisar una captura detallada

Use el control de selección cuando existen varias imágenes. Por defecto se muestra la captura más reciente.

Las cuatro vistas son:

1. **Original:** fotografía cargada.
2. **Filtrado:** imagen suavizada y corregida con CLAHE.
3. **Mapa de relieve:** azul/cian representa superficies lisas; amarillo/rojo representa rugosidad o abolladura.
4. **Vista asistida:** superposición con la silueta detectada por YOLOv8-seg.

El contorno de la silueta aparece en cian.

---

# 9. Interpretación técnica

## 9.1. Características predictivas

El modelo clasificador utiliza las siguientes características:

- Ciclo.
- Temperatura.
- Carga del motor.
- Velocidad de línea.
- Excentricidad.
- Ovalamiento.
- Rugosidad.
- Área.
- Promedios móviles de ovalamiento, rugosidad y temperatura.

## 9.2. Métricas de visión

| Métrica | Interpretación |
|---|---|
| Excentricidad | Desviación del centro respecto al centro ideal |
| Ovalamiento | Diferencia relativa entre los ejes de la elipse |
| Rugosidad | Promedio del gradiente de textura dentro del tapón |
| Área crítica | Porcentaje de superficie detectada como posible daño |
| Índice de rugosidad | Promedio del mapa combinado dentro de la silueta válida |

## 9.3. Criterios de decisión

- `Riesgo de defecto >= 70%` o `RUL <= 5 ciclos`: alerta crítica predictiva.
- `Riesgo de defecto >= 35%`: precaución predictiva.
- Área crítica real por encima del umbral configurado: alerta visual.

Los dos módulos son complementarios. Una alerta visual puede existir aunque el módulo predictivo no marque riesgo crítico, y viceversa.

---

# 10. Reentrenamiento del sistema

El reentrenamiento es una operación técnica y debe realizarse con respaldo de los datos.

## 10.1. Actualizar el conjunto de datos

Edite o sustituya:

```text
data/telemetria_simulada.csv
```

Debe conservar las columnas utilizadas por `src/modelo.py`, en especial `corrida_id`, `ciclo`, `temperatura_c`, `carga_motor_pct`, `velocidad_linea_ms`, `desgaste_real`, `rul_ciclos` y `defecto`.

## 10.2. Reentrenar

Desde la carpeta del proyecto:

```powershell
python src/modelo.py
```

El proceso:

1. Construye las características de visión.
2. Recalcula tendencias móviles.
3. Regenera `data/dataset_completo.csv`.
4. Realiza validación por grupos con 4 corridas.
5. Guarda los modelos nuevos en `data/modelo_riesgo.joblib` y `data/modelo_rul.joblib`.

Después reinicie Streamlit para cargar los archivos actualizados.

## 10.3. Verificar antes de liberar

- Realice pruebas con corridas conocidas.
- Compare las predicciones contra casos de referencia.
- Verifique que no existan valores inválidos o ausentes.
- Conserve una copia de los modelos anteriores.
- Documente la fecha y los datos usados en el entrenamiento.

---

# 11. Procedimiento operativo recomendado

## Revisión normal

1. Inicie la aplicación.
2. Seleccione la corrida activa.
3. Revise el ciclo actual.
4. Confirme que el estado sea normal.
5. Observe las tendencias.
6. Si existe cambio significativo, registre una decisión PDCA.

## Inspección visual

1. Tome la fotografía según las recomendaciones.
2. Revise el estado y el porcentaje de área crítica.
3. Repita la captura si aparece **SIN TAPÓN** o si la imagen es borrosa.
4. Ajuste únicamente la sensibilidad si las condiciones de luz se mantienen.
5. Valide el diagnóstico con una inspección física.
6. Registre la acción correspondiente.

## Ante una alerta crítica

1. Detenga o continúe únicamente según el procedimiento autorizado.
2. Confirme que la imagen no tenga reflejos ni interferencias.
3. Repita la medición con varias capturas.
4. Compare con tendencias predictivas.
5. Solicite inspección de mantenimiento.
6. Registre la decisión y sus observaciones.

---

# 12. Solución de problemas

## La aplicación no inicia

Compruebe:

```powershell
python --version
.\venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

Si falta un módulo, instálelo dentro del entorno virtual:

```powershell
python -m pip install <nombre-del-modulo>
```

No ejecute el proyecto con el entorno global si no es necesario.

## Falta `modelo_riesgo.joblib` o `modelo_rul.joblib`

Ejecute:

```powershell
python src/modelo.py
```

Confirme que ambos archivos existan dentro de `data`.

## Falta `models/best.pt`

El modelo YOLOv8-seg es necesario para la inspección real. Coloque el archivo en:

```text
models/best.pt
```

Sin ese archivo, la aplicación muestra una excepción al procesar la primera imagen real.

## Falta `data/dataset_completo.csv`

Ejecute:

```powershell
python src/modelo.py
```

Esto crea el conjunto de datos y también regenera los modelos.

## No se detecta el tapón

- Acerque o centre el tapón.
- Mejore la iluminación.
- Elimine movimiento y desenfoque.
- Evite fotografiar únicamente el fondo.
- Ajuste manualmente el encuadre.
- Tome otra captura.

## Se detectan demasiadas áreas críticas

- Verifique los reflejos.
- Disminuya ligeramente la sensibilidad.
- Eleve el umbral de alerta si las condiciones de iluminación lo justifican.
- Repita la captura con igual posición e iluminación.
- No cambie los umbrales sin validar los resultados con casos reales.

## Las imágenes no llegan al monitor

- Confirme que celular y PC estén en la misma red.
- Use la URL de red proporcionada por Streamlit.
- Mantenga abierta la aplicación en el navegador.
- Verifique que la opción **Auto-actualizar monitor** esté activa.
- Confirme permisos de cámara y almacenamiento.

## El historial no se guarda

Compruebe permisos de escritura sobre:

```text
data/historial_operador.csv
```

No abra el archivo CSV en Excel mientras Streamlit intenta agregar registros.

## Streamlit muestra datos antiguos

Presione **R** en la interfaz o use el botón para recargar la aplicación. Reiniciar Streamlit también es apropiado después de reemplazar modelos o archivos CSV.

---

# 13. Mantenimiento y respaldo

Se recomienda conservar copias periódicas de:

- `data/dataset_completo.csv`.
- `data/historial_operador.csv`.
- `data/modelo_riesgo.joblib`.
- `data/modelo_rul.joblib`.
- `models/best.pt`.
- `data/capturas_vivo`.

No elimine archivos del proyecto mientras la aplicación esté abierta.

Antes de actualizar el sistema:

1. Detenga Streamlit.
2. Cree una carpeta de respaldo con fecha.
3. Copie los archivos indicados.
4. Instale o reemplaza únicamente los componentes probados.
5. Ejecute una validación básica.

---

# 14. Limitaciones

- Los datos predictivos incluidos pueden ser simulados.
- Las fotografías reales dependen de distancia, iluminación y cámara.
- Los umbrales visuales pueden requerir recalibración en planta.
- Los resultados no sustituyen inspecciones físicas ni protocolos de seguridad.
- El registro PDCA es una herramienta de apoyo y no activa automáticamente acciones industriales.

---

# 15. Archivos y estructura del proyecto

```text
Hackatec Nacional/
├── app.py
├── MANUAL_USUARIO.md
├── src/
│   ├── modelo.py
│   ├── simulador.py
│   └── vision.py
├── data/
│   ├── capturas_vivo/
│   ├── dataset_completo.csv
│   ├── historial_operador.csv
│   ├── modelo_riesgo.joblib
│   ├── modelo_rul.joblib
│   ├── telemetria_simulada.csv
│   └── muestra_tapon.png
├── models/
│   └── best.pt
└── venv/
```
