"""
config.py - Rutas y constantes del proyecto Bank Marketing (UCI)
"""
from pathlib import Path

# Rutas base
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Archivo de datos original
DATASET_NAME = "bank-full.csv"
CSV_SEP = ";"
RAW_DATA_PATH = RAW_DATA_DIR / DATASET_NAME
CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "bank_clean.csv"

# Target y variables (sin 'duration' por fuga de datos)
TARGET = "y"
NUMERIC_FEATURES = [
    "age", "balance", "day", "campaign",
    "pdays", "previous", "contacted_before"
]
CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default",
    "housing", "loan", "contact", "month", "poutcome"
]

# Reproducibilidad y particion (64% train, 16% val, 20% test)
RANDOM_STATE = 42
SEEDS_VARIABILITY = [42, 7, 21, 99, 123]
TEST_SIZE = 0.20
VAL_SIZE = 0.20  # 20% sobre el 80% restante = 16% del total
CV_FOLDS = 5

# Crear directorios si no existen
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)