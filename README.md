# Clasificación Supervisada: Bank Marketing (UCI)

Proyecto final de Ciencia de Datos sobre el conjunto **Bank Marketing** del repositorio UCI (45,211 registros, 16 variables predictoras y variable objetivo binaria `y`), con el objetivo de predecir antes de realizar una llamada si un cliente contratará un depósito a plazo fijo.

## Estructura del proyecto

proyecto_bank_marketing/
├── data/
│   ├── processed/                 # Datos limpios y particiones train / val / test
│   └── raw/
│       └── bank-full.csv          # Dataset original (UCI)
├── models/                        # Preprocesador y modelos entrenados
├── notebooks/
│   └── 01_eda_bank.ipynb          # Notebook de analisis exploratorio de datos (EDA)
├── reports/
│   ├── figures/                   # Graficas en 300 DPI para presentacion y reporte IEEE
│   ├── analisis_errores_detalle.csv
│   ├── analisis_errores_resumen.csv
│   ├── cv_variabilidad_clasicos.csv
│   ├── espacio_busqueda.json
│   ├── mlp_busqueda.csv
│   ├── mlp_mejor_config.json
│   ├── mlp_variabilidad_semillas.csv
│   ├── Reporte.pdf                # Reporte final en PDF formato IEEE
│   ├── reporte_ieee.tex           # Articulo en formato IEEE a dos columnas
│   └── tabla_comparativa_test.csv
├── src/
│   ├── data/
│   │   ├── make_dataset.py        # Carga, validacion y limpieza inicial
│   │   └── preprocess.py          # Split estratificado, One-Hot y StandardScaler
│   ├── evaluation/
│   │   └── evaluate.py            # Metricas en test, curvas ROC/PR y analisis de errores
│   └── models/
│       ├── train_mlp.py           # Entrenamiento, ajuste y semillas del MLP
│       └── train_models.py        # Regresion Logistica, Naive Bayes y SVM (lineal/RBF)
├── .gitignore
├── config.py                      # Rutas, constantes y semillas fijas
└── requirements.txt

1.Cómo ejecutarlo en Visual Studio Code
Abre la carpeta del proyecto en Visual Studio Code.

2.Crea y activa el entorno virtual desde la terminal integrada (Ctrl + ñ o Terminal > New Terminal):

python -m venv bank_env
.\bank_env\Scripts\Activate.ps1
Selecciona el intérprete del entorno (Ctrl + Shift + P > Python: Select Interpreter > .\bank_env\Scripts\python.exe).

3.Instala las dependencias:
pip install -r requirements.txt

4.Ejecuta los scripts en orden desde la raíz del proyecto:
python src/data/make_dataset.py
python src/data/preprocess.py
python src/models/train_models.py
python src/models/train_mlp.py
python src/evaluation/evaluate.py