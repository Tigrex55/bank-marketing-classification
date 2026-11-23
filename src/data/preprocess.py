"""
preprocess.py
Pipeline de preprocesamiento: split estratificado train/val/test,
codificacion One-Hot, escalado estandar y guardado sin fuga de datos.
"""
import pandas as pd
import joblib
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (
    CLEAN_DATA_PATH, PROCESSED_DATA_DIR, MODELS_DIR,
    TARGET, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    RANDOM_STATE, TEST_SIZE, VAL_SIZE
)


def preprocess_and_split(df=None):
    """
    - Carga bank_clean.csv si no se pasa un DataFrame
    - Separa X e y
    - Divide en Train / Val / Test estratificado con semilla fija
    - Ajusta ColumnTransformer (StandardScaler + OneHotEncoder) SOLO en Train
    - Guarda conjuntos procesados y el preprocesador en disco
    """
    if df is None:
        df = pd.read_csv(CLEAN_DATA_PATH)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # 1. Separar conjunto de prueba (20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # 2. Separar entrenamiento (64%) y validacion (16%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_val
    )

    # Guardar X_test sin transformar para el analisis de errores
    X_test.to_csv(PROCESSED_DATA_DIR / "X_test_raw.csv", index=False)

    # 3. Preprocesador ajustado UNICAMENTE con X_train
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)
        ]
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    # Recuperar nombres de columnas finales
    cat_names = list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    )
    feature_names = NUMERIC_FEATURES + cat_names

    X_train_df = pd.DataFrame(X_train_proc, columns=feature_names)
    X_val_df = pd.DataFrame(X_val_proc, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_proc, columns=feature_names)

    # 4. Guardar conjuntos procesados
    X_train_df.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    X_val_df.to_csv(PROCESSED_DATA_DIR / "X_val.csv", index=False)
    X_test_df.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    y_val.to_csv(PROCESSED_DATA_DIR / "y_val.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)

    # Guardar preprocesador
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")

    print(f"Train shape: {X_train_df.shape} | Val shape: {X_val_df.shape} | Test shape: {X_test_df.shape}")
    print(f"Proporcion positivos -> Train: {y_train.mean():.4f} | Val: {y_val.mean():.4f} | Test: {y_test.mean():.4f}")
    print("✓ Preprocesamiento completado y archivos guardados")

    return X_train_df, X_val_df, X_test_df, y_train, y_val, y_test, preprocessor


if __name__ == "__main__":
    preprocess_and_split()