from pathlib import Path
from typing import Dict, List

import numpy as np

APP_TITLE = "Prediksi Risiko Diabetes"
APP_SUBTITLE = "The Intelligence Battle: Human Expert vs. Evolutionary Tuning & Neuro-Fuzzy"


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "diabetes.csv"
MODEL_DIR = BASE_DIR / "model"
MF_PARAMS_PATH = MODEL_DIR / "mf_params.json"
ANFIS_MODEL_PATH = MODEL_DIR / "anfis_model.pkl"

GLUCOSE_RANGE = np.arange(40, 261, 1)
BMI_RANGE = np.arange(10, 71, 0.1)
AGE_RANGE = np.arange(18, 101, 1)
RISK_RANGE = np.arange(0, 1.001, 0.001)

DEFAULT_FEATURE_ORDER = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

CORE_FIS_FEATURES = ["Glucose", "BMI", "Age"]

MF_SETS: Dict[str, List[str]] = {
    "Glucose": ["low", "medium", "high"],
    "BMI": ["low", "medium", "high"],
    "Age": ["low", "medium", "high"],
}
