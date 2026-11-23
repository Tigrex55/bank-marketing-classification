"""
train_mlp.py
Entrenamiento, ajuste de hiperparametros y variabilidad de semillas
para la Red Neuronal (Perceptron Multicapa - MLP) en TensorFlow/Keras.
"""
import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (
    PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR,
    RANDOM_STATE, SEEDS_VARIABILITY
)


def fijar_semillas(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def construir_mlp(n_features, unidades=(64, 32), dropout=0.3, l2_reg=1e-4, lr=0.001):
    """
    Arquitectura MLP para datos tabulares desbalanceados:
    - Capas densas con activacion ReLU y regularizacion L2
    - Dropout para evitar sobreajuste
    - Capa de salida sigmoide para clasificacion binaria
    """
    modelo = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(unidades[0], activation="relu", kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dropout(dropout),
        layers.Dense(unidades[1], activation="relu", kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dropout(dropout / 2),
        layers.Dense(1, activation="sigmoid")
    ])

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    return modelo


def entrenar_mlp():
    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv").values
    y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv").values.ravel()
    X_val = pd.read_csv(PROCESSED_DATA_DIR / "X_val.csv").values
    y_val = pd.read_csv(PROCESSED_DATA_DIR / "y_val.csv").values.ravel()

    n_features = X_train.shape[1]

    # Calcular pesos de clase por el desbalance (~88.3% no / ~11.7% yes)
    pesos = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train)
    class_weights = {0: float(pesos[0]), 1: float(pesos[1])}
    print(f"Pesos de clase calculados: {class_weights}")

    # Espacio de busqueda de hiperparametros sobre el conjunto de validacion
    configuraciones = [
        {"unidades": (64, 32), "dropout": 0.3, "l2_reg": 1e-4, "lr": 0.001},
        {"unidades": (128, 64), "dropout": 0.3, "l2_reg": 1e-4, "lr": 0.001},
        {"unidades": (64, 32), "dropout": 0.2, "l2_reg": 1e-3, "lr": 0.0005},
        {"unidades": (128, 64), "dropout": 0.4, "l2_reg": 1e-3, "lr": 0.0005},
    ]

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    resultados_busqueda = []
    mejor_f1 = -1
    mejor_config = None
    mejor_modelo = None
    mejor_history = None

    print("\n--- Busqueda de hiperparametros MLP ---")
    for i, cfg in enumerate(configuraciones, 1):
        fijar_semillas(RANDOM_STATE)
        modelo = construir_mlp(n_features, **cfg)
        history = modelo.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=80,
            batch_size=128,
            class_weight=class_weights,
            callbacks=[early_stop],
            verbose=0
        )

        y_val_prob = modelo.predict(X_val, verbose=0).ravel()
        y_val_pred = (y_val_prob >= 0.5).astype(int)
        f1 = f1_score(y_val, y_val_pred)
        roc = roc_auc_score(y_val, y_val_prob)
        pr = average_precision_score(y_val, y_val_prob)

        print(f"  Config {i} {cfg} -> F1 Val: {f1:.4f} | ROC: {roc:.4f} | PR-AUC: {pr:.4f} | Epocas: {len(history.history['loss'])}")
        resultados_busqueda.append({
            **cfg,
            "unidades": str(cfg["unidades"]),
            "f1_val": round(f1, 4),
            "roc_auc_val": round(roc, 4),
            "pr_auc_val": round(pr, 4),
            "epocas": len(history.history["loss"])
        })

        if f1 > mejor_f1:
            mejor_f1 = f1
            mejor_config = cfg
            mejor_modelo = modelo
            mejor_history = history

    # Guardar mejor modelo y resumen
    mejor_modelo.save(MODELS_DIR / "mlp_model.keras")
    pd.DataFrame(resultados_busqueda).to_csv(REPORTS_DIR / "mlp_busqueda.csv", index=False)

    with open(REPORTS_DIR / "mlp_mejor_config.json", "w", encoding="utf-8") as f:
        json.dump(mejor_config, f, indent=4)

    # Grafica de entrenamiento (estilo IEEE: 2 paneles limpios)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(mejor_history.history["loss"], label="Entrenamiento")
    axes[0].plot(mejor_history.history["val_loss"], label="Validacion")
    axes[0].set_title("Perdida (Binary Crossentropy)")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(mejor_history.history["auc"], label="Entrenamiento")
    axes[1].plot(mejor_history.history["val_auc"], label="Validacion")
    axes[1].set_title("ROC-AUC durante el entrenamiento")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("AUC")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mlp_entrenamiento.png", dpi=300)
    plt.close()

    # Evaluacion de variabilidad con 5 semillas fijas usando la mejor arquitectura
    print(f"\n--- Evaluando variabilidad del MLP ({len(SEEDS_VARIABILITY)} semillas) ---")
    filas_semillas = []
    for seed in SEEDS_VARIABILITY:
        fijar_semillas(seed)
        m = construir_mlp(n_features, **mejor_config)
        m.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=80,
            batch_size=128,
            class_weight=class_weights,
            callbacks=[early_stop],
            verbose=0
        )
        probs = m.predict(X_val, verbose=0).ravel()
        preds = (probs >= 0.5).astype(int)
        filas_semillas.append({
            "seed": seed,
            "accuracy": accuracy_score(y_val, preds),
            "precision": precision_score(y_val, preds),
            "recall": recall_score(y_val, preds),
            "f1": f1_score(y_val, preds),
            "roc_auc": roc_auc_score(y_val, probs),
            "pr_auc": average_precision_score(y_val, probs)
        })

    df_semillas = pd.DataFrame(filas_semillas)
    df_semillas.to_csv(REPORTS_DIR / "mlp_variabilidad_semillas.csv", index=False)

    print("MLP Variabilidad en Validacion (Media +/- Std):")
    for col in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        print(f"  {col:10s}: {df_semillas[col].mean():.4f} +/- {df_semillas[col].std():.4f}")

    print("\n✓ MLP entrenado y guardado en models/mlp_model.keras")


if __name__ == "__main__":
    entrenar_mlp()