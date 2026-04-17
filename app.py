import streamlit as st

from diabetes_app.config import APP_SUBTITLE, APP_TITLE, DATASET_PATH
from diabetes_app.loaders import load_anfis_model, load_dataset, load_mf_params
from diabetes_app.theme import configure_page, inject_theme

try:
    import skfuzzy as _skfuzzy  # noqa: F401
except ImportError:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.error("Library scikit-fuzzy belum terpasang. Jalankan: pip install -r requirements.txt")
    st.stop()

from diabetes_app.pages import render_home_tab, render_predict_tab, render_shift_tab

configure_page()
inject_theme()

st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

dataset, dataset_error = load_dataset(str(DATASET_PATH))
mf_payload, mf_error = load_mf_params()
anfis_model, anfis_model_error = load_anfis_model()

home_errors = [msg for msg in [dataset_error, mf_error] if msg]
predict_errors = [msg for msg in [mf_error] if msg]
shift_errors = [msg for msg in [mf_error] if msg]

tab_predict, tab_home, tab_shift = st.tabs(
    [
        "Simulator Prediksi",
        "EDA",
        "Analisis Kurva MF",
    ]
)

with tab_home:
    render_home_tab(dataset, home_errors, mf_payload, anfis_model_error)

with tab_predict:
    render_predict_tab(mf_payload, anfis_model, predict_errors, anfis_model_error)

with tab_shift:
    render_shift_tab(mf_payload, shift_errors)
