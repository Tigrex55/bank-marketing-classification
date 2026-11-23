"""
eda_figures.py
Genera las graficas del Analisis Exploratorio de Datos (EDA) en 300 DPI
para la presentacion y el reporte en formato IEEE.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import RAW_DATA_PATH, FIGURES_DIR, CSV_SEP

sns.set(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (9, 5)


def generar_graficas_eda():
    df = pd.read_csv(RAW_DATA_PATH, sep=CSV_SEP)
    df["y_bin"] = (df["y"] == "yes").astype(int)

    # 1. Desbalance de clases (Fig. 1 del articulo IEEE)
    plt.figure(figsize=(7, 4.5))
    ax = sns.countplot(x="y", data=df)
    plt.title("Distribucion de la variable objetivo (Contratacion de deposito)")
    plt.xlabel("Contrato deposito a plazo (y)")
    plt.ylabel("Cantidad de clientes")
    total = len(df)
    for p in ax.patches:
        porcentaje = f"{100 * p.get_height() / total:.2f}%"
        ax.annotate(
            porcentaje,
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center", va="bottom", fontweight="bold"
        )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_desbalance_clases.png", dpi=300)
    plt.close()

    # 2. Faltantes implicitos ('unknown')
    cat_cols = [c for c in df.columns if df[c].dtype == "object" and c != "y"]
    unk = {c: round((df[c] == "unknown").mean() * 100, 2) for c in cat_cols if (df[c] == "unknown").sum() > 0}
    df_unk = pd.DataFrame({"variable": list(unk.keys()), "porcentaje": list(unk.values())}).sort_values("porcentaje", ascending=False)

    plt.figure(figsize=(7, 4.5))
    ax = sns.barplot(x="variable", y="porcentaje", data=df_unk)
    plt.title("Porcentaje de valores 'unknown' por variable categorica")
    plt.xlabel("Variable")
    plt.ylabel("Porcentaje (%)")
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}%",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center", va="bottom", fontweight="bold"
        )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_valores_unknown.png", dpi=300)
    plt.close()

    # 3. Distribucion de variables numericas clave
    num_vars = ["age", "balance", "duration", "campaign"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, col in zip(axes.flatten(), num_vars):
        sns.histplot(df[col], kde=True, bins=30, ax=ax)
        ax.set_title(f"Distribucion de {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_distribucion_numericas.png", dpi=300)
    plt.close()

    # 4. Analisis bivariado (Fig. 2 del articulo IEEE)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    sns.barplot(x="poutcome", y="y_bin", data=df, ax=axes[0], errorbar=None)
    axes[0].set_title("Aceptacion por poutcome")
    axes[0].set_ylabel("Proporcion de 'yes'")

    sns.barplot(x="housing", y="y_bin", data=df, ax=axes[1], errorbar=None)
    axes[1].set_title("Aceptacion por hipoteca")
    axes[1].set_ylabel("")

    orden_job = df.groupby("job")["y_bin"].mean().sort_values(ascending=False).index
    sns.barplot(y="job", x="y_bin", data=df, order=orden_job, ax=axes[2], errorbar=None)
    axes[2].set_title("Aceptacion por ocupacion")
    axes[2].set_xlabel("Proporcion de 'yes'")
    axes[2].set_ylabel("")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_analisis_bivariado.png", dpi=300)
    plt.close()

    # 5. Mapa de correlaciones
    num_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous", "y_bin"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Mapa de correlaciones (Variables numericas y objetivo)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_mapa_correlaciones.png", dpi=300)
    plt.close()

    print("✓ 5 graficas de EDA guardadas en reports/figures/")


if __name__ == "__main__":
    generar_graficas_eda()