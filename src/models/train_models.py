"""
train_models.py
Entrenamiento y ajuste de hiperparametros con validacion cruzada para:
- Modelo de referencia: Regresion Logistica
- Clasificador Bayesiano: Gaussian Naive Bayes (y comparacion con BernoulliNB)
- SVM: Kernel Lineal (LinearSVC) y Kernel RBF (SVC)
"""
import json
import time
import joblib
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import LinearSVC, SVC

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, RANDOM_STATE, CV_FOLDS


def cargar_datos():
    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv").values.ravel()
    X_val = pd.read_csv(PROCESSED_DATA_DIR / "X_val.csv")
    y_val = pd.read_csv(PROCESSED_DATA_DIR / "y_val.csv").values.ravel()
    return X_train, y_train, X_val, y_val


def evaluar_variabilidad_cv(nombre, modelo, X, y, cv):
    """Calcula media y desviacion estandar entre folds para reportar variabilidad."""
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision"
    }
    res = cross_validate(modelo, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    fila = {"modelo": nombre}
    for m in scoring:
        fila[f"{m}_mean"] = round(res[f"test_{m}"].mean(), 4)
        fila[f"{m}_std"] = round(res[f"test_{m}"].std(), 4)
    return fila


def entrenar_modelos():
    X_train, y_train, _, _ = cargar_datos()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    espacio_busqueda = {}
    resumen_cv = []

    # ---------------------------------------------------------
    # 1. Modelo de referencia: Regresion Logistica
    # ---------------------------------------------------------
    print("\n[1/4] Ajustando Regresion Logistica (modelo de referencia)...")
    t0 = time.time()
    grid_lr = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "class_weight": [None, "balanced"]
    }
    gs_lr = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        param_grid=grid_lr,
        scoring="f1",
        cv=cv,
        n_jobs=-1
    )
    gs_lr.fit(X_train, y_train)
    best_lr = gs_lr.best_estimator_
    joblib.dump(best_lr, MODELS_DIR / "logistic_regression.joblib")
    espacio_busqueda["Regresion_Logistica"] = {
        "espacio": grid_lr,
        "mejores_params": gs_lr.best_params_,
        "mejor_f1_cv": round(gs_lr.best_score_, 4)
    }
    resumen_cv.append(evaluar_variabilidad_cv("Regresion Logistica", best_lr, X_train, y_train, cv))
    print(f"  Mejores params: {gs_lr.best_params_} | F1 CV: {gs_lr.best_score_:.4f} ({time.time()-t0:.1f}s)")

    # ---------------------------------------------------------
    # 2. Clasificador Bayesiano: Naive Bayes
    # ---------------------------------------------------------
    print("\n[2/4] Ajustando Naive Bayes (GaussianNB y prueba con BernoulliNB)...")
    t0 = time.time()
    grid_gnb = {"var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-2, 1e-1]}
    gs_gnb = GridSearchCV(
        GaussianNB(),
        param_grid=grid_gnb,
        scoring="f1",
        cv=cv,
        n_jobs=-1
    )
    gs_gnb.fit(X_train, y_train)

    # Comparacion auxiliar con BernoulliNB para justificar eleccion
    grid_bnb = {"alpha": [0.1, 0.5, 1.0, 2.0]}
    gs_bnb = GridSearchCV(
        BernoulliNB(),
        param_grid=grid_bnb,
        scoring="f1",
        cv=cv,
        n_jobs=-1
    )
    gs_bnb.fit(X_train, y_train)
    print(f"  GaussianNB F1 CV: {gs_gnb.best_score_:.4f} | BernoulliNB F1 CV: {gs_bnb.best_score_:.4f}")

    best_nb = gs_gnb.best_estimator_
    joblib.dump(best_nb, MODELS_DIR / "naive_bayes.joblib")
    espacio_busqueda["Naive_Bayes_Gaussian"] = {
        "espacio": grid_gnb,
        "mejores_params": gs_gnb.best_params_,
        "mejor_f1_cv": round(gs_gnb.best_score_, 4),
        "comparacion_bernoulli_f1_cv": round(gs_bnb.best_score_, 4)
    }
    resumen_cv.append(evaluar_variabilidad_cv("Naive Bayes (Gaussian)", best_nb, X_train, y_train, cv))
    print(f"  Mejores params GNB: {gs_gnb.best_params_} ({time.time()-t0:.1f}s)")

    # ---------------------------------------------------------
    # 3. SVM con Kernel Lineal
    # ---------------------------------------------------------
    print("\n[3/4] Ajustando SVM Lineal...")
    t0 = time.time()
    grid_svm_lin = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "class_weight": [None, "balanced"]
    }
    gs_svm_lin = GridSearchCV(
        LinearSVC(dual=False, max_iter=3000, random_state=RANDOM_STATE),
        param_grid=grid_svm_lin,
        scoring="f1",
        cv=cv,
        n_jobs=-1
    )
    gs_svm_lin.fit(X_train, y_train)
    best_svm_lin = gs_svm_lin.best_estimator_
    joblib.dump(best_svm_lin, MODELS_DIR / "svm_linear.joblib")
    espacio_busqueda["SVM_Lineal"] = {
        "espacio": grid_svm_lin,
        "mejores_params": gs_svm_lin.best_params_,
        "mejor_f1_cv": round(gs_svm_lin.best_score_, 4)
    }
    resumen_cv.append(evaluar_variabilidad_cv("SVM Lineal", best_svm_lin, X_train, y_train, cv))
    print(f"  Mejores params: {gs_svm_lin.best_params_} | F1 CV: {gs_svm_lin.best_score_:.4f} ({time.time()-t0:.1f}s)")

    # ---------------------------------------------------------
    # 4. SVM con Kernel RBF (busqueda en submuestra estratificada + ajuste total)
    # ---------------------------------------------------------
    print("\n[4/4] Ajustando SVM RBF (C y gamma)...")
    t0 = time.time()
    X_sub, _, y_sub, _ = train_test_split(
        X_train, y_train,
        train_size=8000,
        random_state=RANDOM_STATE,
        stratify=y_train
    )
    grid_svm_rbf = {
        "C": [0.1, 1.0, 10.0],
        "gamma": ["scale", 0.01, 0.1],
        "class_weight": ["balanced"]
    }
    gs_svm_rbf = GridSearchCV(
        SVC(kernel="rbf", cache_size=1000, random_state=RANDOM_STATE),
        param_grid=grid_svm_rbf,
        scoring="f1",
        cv=cv,
        n_jobs=-1
    )
    gs_svm_rbf.fit(X_sub, y_sub)
    print(f"  Mejores params en submuestra (8,000): {gs_svm_rbf.best_params_} | F1 CV: {gs_svm_rbf.best_score_:.4f}")

    # Reentrenar el modelo ganador sobre TODO X_train
    best_svm_rbf = SVC(
        kernel="rbf",
        cache_size=1000,
        random_state=RANDOM_STATE,
        **gs_svm_rbf.best_params_
    )
    best_svm_rbf.fit(X_train, y_train)
    joblib.dump(best_svm_rbf, MODELS_DIR / "svm_rbf.joblib")

    espacio_busqueda["SVM_RBF"] = {
        "espacio": grid_svm_rbf,
        "submuestra_tuning": 8000,
        "mejores_params": gs_svm_rbf.best_params_,
        "mejor_f1_cv_submuestra": round(gs_svm_rbf.best_score_, 4)
    }
    # Variabilidad CV de SVM RBF (evaluada en la submuestra estratificada de 8,000 para rapidez)
    resumen_cv.append(evaluar_variabilidad_cv("SVM RBF", best_svm_rbf, X_sub, y_sub, cv))
    print(f"  SVM RBF reentrenado en todo X_train ({time.time()-t0:.1f}s)")

    # Guardar reportes de CV y espacio de busqueda
    df_cv = pd.DataFrame(resumen_cv)
    df_cv.to_csv(REPORTS_DIR / "cv_variabilidad_clasicos.csv", index=False)

    with open(REPORTS_DIR / "espacio_busqueda.json", "w", encoding="utf-8") as f:
        json.dump(espacio_busqueda, f, indent=4, ensure_ascii=False)

    print("\n--- Resumen de Validacion Cruzada (Media +/- Std) ---")
    print(df_cv[["modelo", "f1_mean", "f1_std", "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "pr_auc_std"]])
    print("\n✓ Modelos clasicos entrenados y guardados en models/")


if __name__ == "__main__":
    entrenar_modelos()