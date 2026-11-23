"""
make_dataset.py
Carga el dataset original, realiza validaciones basicas y limpieza inicial.
"""
import pandas as pd
import sys
from pathlib import Path

# Anadir la raiz del proyecto al path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import RAW_DATA_PATH, CLEAN_DATA_PATH, CSV_SEP, TARGET


def load_raw_data(path=RAW_DATA_PATH):
    """Carga el CSV original y muestra informacion basica."""
    print(f"Cargando datos desde: {path}")
    df = pd.read_csv(path, sep=CSV_SEP)
    print(f"Shape original: {df.shape}")
    print(f"Columnas: {list(df.columns)}")
    print(f"\nDistribucion del target ({TARGET}):")
    print(df[TARGET].value_counts())
    print(df[TARGET].value_counts(normalize=True).round(4))
    print(f"\nValores nulos (NaN): {df.isnull().sum().sum()}")
    print(f"Filas duplicadas: {df.duplicated().sum()}")

    print("\nValores 'unknown' por columna categorica:")
    for col in df.columns:
        if df[col].dtype == "object" or str(df[col].dtype) == "str":
            unk = (df[col] == "unknown").sum()
            if unk > 0:
                print(f"  - {col}: {unk} ({unk / len(df) * 100:.2f}%)")
    return df


def clean_data(df):
    """
    Limpieza documentada del dataset:
    - Elimina 'duration' (solo se conoce tras finalizar la llamada -> data leakage).
    - Crea indicador binario 'contacted_before' a partir de pdays != -1
      y reemplaza -1 por 0 en pdays para no distorsionar el escalado.
    - Conserva 'unknown' como categoria valida (faltante informativo).
    - Codifica el target 'y' a binario (yes=1, no=0).
    """
    df_clean = df.copy()

    if "duration" in df_clean.columns:
        df_clean = df_clean.drop(columns=["duration"])
        print("\nColumna 'duration' eliminada para evitar fuga de datos.")

    df_clean["contacted_before"] = (df_clean["pdays"] != -1).astype(int)
    df_clean["pdays"] = df_clean["pdays"].replace(-1, 0)

    df_clean[TARGET] = (df_clean[TARGET] == "yes").astype(int)
    print(f"Shape despues de limpieza: {df_clean.shape}")
    return df_clean


def save_processed(df, out_path=CLEAN_DATA_PATH):
    """Guarda la version limpia intermedia."""
    df.to_csv(out_path, index=False)
    print(f"Guardado en: {out_path}")
    return out_path


if __name__ == "__main__":
    df_raw = load_raw_data()
    df_clean = clean_data(df_raw)
    save_processed(df_clean)
    print("\nmake_dataset.py ejecutado correctamente")