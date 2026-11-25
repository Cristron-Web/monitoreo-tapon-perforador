import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, mean_absolute_error

from src.simulador import generar_corridas, generar_cuadro_tapon
from src.vision import analizar_cuadro

FEATURES = [
    "ciclo",
    "temperatura_c",
    "carga_motor_pct",
    "velocidad_linea_ms",
    "excentricidad",
    "ovalamiento",
    "rugosidad",
    "area_px",
    "ovalamiento_ma3",
    "rugosidad_ma3",
    "temp_ma3",
]


def construir_dataset_completo() -> pd.DataFrame:
    """Combina telemetría + métricas extraídas por OpenCV + tendencias móviles."""
    ruta_csv = "data/telemetria_simulada.csv"
    if os.path.exists(ruta_csv):
        df = pd.read_csv(ruta_csv)
    else:
        df = generar_corridas()

    metricas_vision = []
    for idx, fila in df.iterrows():
        img = generar_cuadro_tapon(desgaste=float(fila["desgaste_real"]), seed=int(idx))
        res = analizar_cuadro(img)
        metricas_vision.append(
            {
                "excentricidad": res["excentricidad"],
                "ovalamiento": res["ovalamiento"],
                "rugosidad": res["rugosidad"],
                "area_px": res["area_px"],
            }
        )

    df_vision = pd.DataFrame(metricas_vision)
    df_full = pd.concat([df.reset_index(drop=True), df_vision], axis=1)

    # Tendencias móviles (ventana de 3 piezas) calculadas dentro de cada corrida
    df_full["ovalamiento_ma3"] = (
        df_full.groupby("corrida_id")["ovalamiento"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )
    df_full["rugosidad_ma3"] = (
        df_full.groupby("corrida_id")["rugosidad"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )
    df_full["temp_ma3"] = (
        df_full.groupby("corrida_id")["temperatura_c"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )

    df_full.to_csv("data/dataset_completo.csv", index=False)
    return df_full


def entrenar_modelos():
    print("Extrayendo características de visión para todas las corridas...")
    df = construir_dataset_completo()

    X = df[FEATURES]
    y_cls = df["defecto"]
    y_rul = df["rul_ciclos"]
    grupos = df["corrida_id"]

    # Validación cruzada agrupada por corrida (sin fuga de datos)
    gkf = GroupKFold(n_splits=4)
    aucs, maes = [], []

    for train_idx, val_idx in gkf.split(X, y_cls, grupos):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]

        clf_cv = lgb.LGBMClassifier(
            n_estimators=80, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1
        )
        clf_cv.fit(X_tr, y_cls.iloc[train_idx])
        probs = clf_cv.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_cls.iloc[val_idx], probs))

        reg_cv = lgb.LGBMRegressor(
            n_estimators=80, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1
        )
        reg_cv.fit(X_tr, y_rul.iloc[train_idx])
        preds_rul = reg_cv.predict(X_val)
        maes.append(mean_absolute_error(y_rul.iloc[val_idx], preds_rul))

    print(f"Validación GroupKFold (4 folds por corrida):")
    print(f"  -> Clasificación Riesgo (ROC-AUC promedio): {np.mean(aucs):.4f}")
    print(f"  -> Regresión Vida Útil  (MAE promedio)    : {np.mean(maes):.2f} ciclos")

    # Entrenamiento final con todo el dataset
    modelo_riesgo = lgb.LGBMClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1
    )
    modelo_riesgo.fit(X, y_cls)

    modelo_rul = lgb.LGBMRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1
    )
    modelo_rul.fit(X, y_rul)

    os.makedirs("data", exist_ok=True)
    joblib.dump(modelo_riesgo, "data/modelo_riesgo.joblib")
    joblib.dump(modelo_rul, "data/modelo_rul.joblib")
    print("Modelos guardados en data/modelo_riesgo.joblib y data/modelo_rul.joblib")


if __name__ == "__main__":
    entrenar_modelos()