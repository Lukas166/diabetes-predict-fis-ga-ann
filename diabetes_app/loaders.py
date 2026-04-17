import json
import pickle
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

from .config import ANFIS_MODEL_PATH, MF_PARAMS_PATH


@st.cache_data(show_spinner=False)
def load_dataset(csv_path: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        dataframe = pd.read_csv(csv_path)
        return dataframe, None
    except FileNotFoundError:
        return None, f"File dataset tidak ditemukan: {csv_path}"
    except Exception as exc:
        return None, f"Gagal membaca dataset: {exc}"


@st.cache_resource(show_spinner=False)
def load_mf_params() -> Tuple[Optional[dict], Optional[str]]:
    try:
        with open(MF_PARAMS_PATH, "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        return payload, None
    except FileNotFoundError:
        return None, f"File parameter tidak ditemukan: {MF_PARAMS_PATH}"
    except json.JSONDecodeError as exc:
        return None, f"Format JSON tidak valid ({MF_PARAMS_PATH}): {exc}"
    except Exception as exc:
        return None, f"Gagal memuat JSON ({MF_PARAMS_PATH}): {exc}"


@st.cache_resource(show_spinner=False)
def load_anfis_model() -> Tuple[Optional[object], Optional[str]]:
    try:
        with open(ANFIS_MODEL_PATH, "rb") as file_obj:
            payload = pickle.load(file_obj)
        return payload, None
    except FileNotFoundError:
        return None, f"File ANFIS model tidak ditemukan: {ANFIS_MODEL_PATH}"
    except Exception as exc:
        return None, f"Gagal memuat ANFIS model ({ANFIS_MODEL_PATH}): {exc}"
