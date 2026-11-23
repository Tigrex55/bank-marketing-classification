# Clasificación Supervisada: Bank Marketing (UCI)

Proyecto final de Ciencia de Datos sobre el conjunto **Bank Marketing** del repositorio UCI (45,211 registros, 16 variables predictoras y variable objetivo binaria `y`), con el objetivo de predecir antes de realizar una llamada si un cliente contratará un depósito a plazo fijo.

1.Cómo ejecutarlo en Visual Studio Code

Abre la carpeta del proyecto en Visual Studio Code.

2.Crea y activa el entorno virtual desde la terminal integrada (Ctrl + ñ o Terminal > New Terminal):

python -m venv bank_env
.\bank_env\Scripts\Activate.ps1
Selecciona el intérprete del entorno (Ctrl + Shift + P > Python: Select Interpreter > .\bank_env\Scripts\python.exe).

3.Instala las dependencias:

pip install -r requirements.txt

4.Ejecuta los scripts en el siguiente orden

python src/data/make_dataset.py
python src/data/preprocess.py
python src/models/train_models.py
python src/models/train_mlp.py
python src/evaluation/evaluate.py