"""
evaluate.py
Evaluacion final en el conjunto de prueba, graficas comparativas (300 DPI para IEEE)
y analisis de errores (falsos positivos y falsos negativos).
"""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve
)

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR

sns.set(style="whitegrid", palette="muted")


def obtener_predicciones(modelo, X, es_keras=False):
    """Devuelve clases predichas (0/1) y puntaje continuo para ROC y PR."""
    if es_keras:
        scores = modelo.predict(X.values, verbose=0).ravel()
        preds = (scores >= 0.5).astype(int)
        return preds, scores

    preds = modelo.predict(X)
    if hasattr(modelo, "predict_proba"):
        scores = modelo.predict_proba(X)[:, 1]
    else:
        scores = modelo.decision_function(X)
    return preds, scores


def evaluar_todos():
    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").values.ravel()
    X_test_raw = pd.read_csv(PROCESSED_DATA_DIR / "X_test_raw.csv")

    modelos = {
        "Regresion Logistica": (joblib.load(MODELS_DIR / "logistic_regression.joblib"), False),
        "Naive Bayes (Gaussian)": (joblib.load(MODELS_DIR / "naive_bayes.joblib"), False),
        "SVM Lineal": (joblib.load(MODELS_DIR / "svm_linear.joblib"), False),
        "SVM RBF": (joblib.load(MODELS_DIR / "svm_rbf.joblib"), False),
        "Red Neuronal (MLP)": (keras.models.load_model(MODELS_DIR / "mlp_model.keras"), True),
    }

    resultados = []
    curvas = {}
    matrices = {}
    predicciones_dict = {}

    for nombre, (modelo, es_keras) in modelos.items():
        y_pred, y_score = obtener_predicciones(modelo, X_test, es_keras=es_keras)
        predicciones_dict[nombre] = (y_pred, y_score)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_score)
        pr_auc = average_precision_score(y_test, y_score)

        resultados.append({
            "Modelo": nombre,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
            "PR-AUC": round(pr_auc, 4)
        })

        matrices[nombre] = confusion_matrix(y_test, y_pred)
        curvas[nombre] = {
            "roc": roc_curve(y_test, y_score),
            "pr": precision_recall_curve(y_test, y_score),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc
        }

    # 1. Tabla comparativa final en test
    df_res = pd.DataFrame(resultados).sort_values("F1-Score", ascending=False)
    df_res.to_csv(REPORTS_DIR / "tabla_comparativa_test.csv", index=False)
    print("\n=== TABLA COMPARATIVA EN CONJUNTO DE PRUEBA ===")
    print(df_res.to_string(index=False))

    # 2. Grafica de Curvas ROC y Precision-Recall lado a lado (lista para IEEE)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for nombre, data in curvas.items():
        fpr, tpr, _ = data["roc"]
        axes[0].plot(fpr, tpr, label=f"{nombre} (AUC = {data['roc_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Azar (0.500)")
    axes[0].set_title("Curvas ROC (Conjunto de Prueba)")
    axes[0].set_xlabel("Tasa de Falsos Positivos (FPR)")
    axes[0].set_ylabel("Tasa de Verdaderos Positivos (Recall)")
    axes[0].legend(fontsize=8, loc="lower right")

    baseline_pr = y_test.mean()
    for nombre, data in curvas.items():
        prec_c, rec_c, _ = data["pr"]
        axes[1].plot(rec_c, prec_c, label=f"{nombre} (AP = {data['pr_auc']:.3f})")
    axes[1].axhline(baseline_pr, color="k", linestyle="--", alpha=0.5, label=f"Base ({baseline_pr:.3f})")
    axes[1].set_title("Curvas Precision-Recall (Conjunto de Prueba)")
    axes[1].set_xlabel("Sensibilidad (Recall)")
    axes[1].set_ylabel("Precision")
    axes[1].legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "curvas_roc_pr.png", dpi=300)
    plt.close()

    # 3. Grafica de Matrices de Confusion
    fig, axes = plt.subplots(1, len(matrices), figsize=(18, 3.6))
    for ax, (nombre, cm) in zip(axes, matrices.items()):
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["No", "Si"], yticklabels=["No", "Si"]
        )
        ax.set_title(nombre, fontsize=10)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "matrices_confusion.png", dpi=300)
    plt.close()

    # 4. Analisis de errores sobre el mejor modelo (por F1-Score)
    mejor_nombre = df_res.iloc[0]["Modelo"]
    y_pred_best, _ = predicciones_dict[mejor_nombre]

    df_err = X_test_raw.copy()
    df_err["y_real"] = y_test
    df_err["y_pred"] = y_pred_best

    condiciones = [
        (df_err["y_real"] == 1) & (df_err["y_pred"] == 1),
        (df_err["y_real"] == 0) & (df_err["y_pred"] == 0),
        (df_err["y_real"] == 0) & (df_err["y_pred"] == 1),
        (df_err["y_real"] == 1) & (df_err["y_pred"] == 0),
    ]
    etiquetas = ["TP (Acierto Si)", "TN (Acierto No)", "FP (Falso Positivo)", "FN (Falso Negativo)"]
    df_err["tipo_resultado"] = np.select(condiciones, etiquetas, default="Otro")

    resumen_errores = df_err.groupby("tipo_resultado")[
        ["age", "balance", "campaign", "previous", "contacted_before"]
    ].agg(["count", "mean", "median"]).round(2)

    print(f"\n=== ANALISIS DE ERRORES ({mejor_nombre}) ===")
    print(df_err["tipo_resultado"].value_counts())
    print("\nMedianas por tipo de resultado:")
    print(df_err.groupby("tipo_resultado")[["age", "balance", "campaign", "previous"]].median().round(2))
    print("\nDistribucion de 'poutcome' por tipo de resultado (%):")
    print(pd.crosstab(df_err["tipo_resultado"], df_err["poutcome"], normalize="index").round(3) * 100)
    print("\nDistribucion de 'housing' por tipo de resultado (%):")
    print(pd.crosstab(df_err["tipo_resultado"], df_err["housing"], normalize="index").round(3) * 100)

    df_err.to_csv(REPORTS_DIR / "analisis_errores_detalle.csv", index=False)
    resumen_errores.to_csv(REPORTS_DIR / "analisis_errores_resumen.csv")
    print("\n✓ Evaluacion completa. Figuras y tablas guardadas en reports/")


if __name__ == "__main__":
    evaluar_todos()