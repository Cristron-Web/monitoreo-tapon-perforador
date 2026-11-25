import os
import cv2
import numpy as np
import pandas as pd


def generar_cuadro_tapon(desgaste: float, seed: int | None = None) -> np.ndarray:
    """
    Simula un cuadro de video (256x256, escala de grises) del tapón perforador.
    desgaste: valor entre 0.0 (nuevo) y 1.0 (falla crítica).
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((256, 256), dtype=np.uint8)

    # Fondo con ruido térmico base
    fondo = rng.normal(loc=35, scale=8, size=(256, 256))
    img = np.clip(fondo, 0, 255).astype(np.uint8)

    # Geometría del tapón: a mayor desgaste, mayor ovalamiento y excentricidad
    centro_x = int(128 + rng.normal(0, 2 + 6 * desgaste))
    centro_y = int(128 + rng.normal(0, 2 + 6 * desgaste))
    eje_mayor = int(70 - 6 * desgaste)
    eje_menor = int(70 - 22 * desgaste)  # Pierde circularidad (ovalamiento)
    angulo = float(rng.uniform(0, 180))

    # Dibuja el cuerpo del tapón (más brillante por alta temperatura)
    intensidad = int(175 + 45 * desgaste)
    cv2.ellipse(
        img,
        (centro_x, centro_y),
        (max(10, eje_mayor), max(10, eje_menor)),
        angulo,
        0,
        360,
        intensidad,
        -1,
    )

    # Simulación de rugosidad / picaduras superficiales por desgaste
    num_defectos = int(5 + 45 * desgaste)
    for _ in range(num_defectos):
        dx = int(rng.normal(0, eje_mayor * 0.45))
        dy = int(rng.normal(0, eje_menor * 0.45))
        radio = int(rng.integers(2, 6))
        tono = int(rng.integers(60, 130))
        cv2.circle(img, (centro_x + dx, centro_y + dy), radio, tono, -1)

    # Desenfoque por movimiento horizontal (alta velocidad de línea)
    kernel_motion = np.zeros((5, 5), dtype=np.float32)
    kernel_motion[2, :] = 1.0 / 5.0
    img = cv2.filter2D(img, -1, kernel_motion)

    # Ruido térmico final
    ruido_termico = rng.normal(0, 6 + 10 * desgaste, img.shape)
    img = np.clip(img.astype(np.float32) + ruido_termico, 0, 255).astype(np.uint8)
    return img


def generar_corridas(n_corridas: int = 8, ciclos_por_corrida: int = 40) -> pd.DataFrame:
    """
    Genera telemetría operativa por corrida y guarda un dataset base en data/.
    """
    rng = np.random.default_rng(42)
    registros = []

    for corrida in range(1, n_corridas + 1):
        # Cada corrida tiene una tasa de desgaste ligeramente distinta
        tasa_corrida = rng.uniform(0.85, 1.20)
        vida_total = int(ciclos_por_corrida / tasa_corrida)

        for ciclo in range(1, ciclos_por_corrida + 1):
            desgaste = min(1.0, (ciclo / vida_total) + rng.normal(0, 0.03))
            desgaste = max(0.0, desgaste)

            temperatura = 920 + 110 * desgaste + rng.normal(0, 8)
            carga_motor = 62 + 28 * desgaste + rng.normal(0, 2.5)
            velocidad_linea = 4.2 - 0.5 * desgaste + rng.normal(0, 0.1)
            rul_ciclos = max(0, vida_total - ciclo)
            defecto = 1 if desgaste >= 0.72 else 0

            registros.append(
                {
                    "corrida_id": corrida,
                    "ciclo": ciclo,
                    "temperatura_c": round(temperatura, 2),
                    "carga_motor_pct": round(carga_motor, 2),
                    "velocidad_linea_ms": round(velocidad_linea, 3),
                    "desgaste_real": round(desgaste, 4),
                    "rul_ciclos": rul_ciclos,
                    "defecto": defecto,
                }
            )

    df = pd.DataFrame(registros)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/telemetria_simulada.csv", index=False)
    return df


if __name__ == "__main__":
    df = generar_corridas()
    img_demo = generar_cuadro_tapon(desgaste=0.75, seed=7)
    cv2.imwrite("data/muestra_tapon.png", img_demo)
    print(f"Dataset generado: {len(df)} registros en data/telemetria_simulada.csv")
    print("Imagen de muestra guardada en data/muestra_tapon.png")