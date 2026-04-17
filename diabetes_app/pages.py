from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from .config import DEFAULT_FEATURE_ORDER
from .inference import anfis_predict_risk, fuzzy_predict_risk, risk_label
from .visuals import plot_eda_charts, plot_performance_bars, plot_shift_analysis, render_prediction_card


def _render_errors(messages: List[str], prefix: str) -> None:
    if messages:
        st.error(prefix)
        for message in messages:
            st.write(f"- {message}")


def _feature_range(mf_payload: dict, feature: str, default_low: float, default_high: float):
    manual_params = (mf_payload or {}).get("manual_mf_params", {})
    feature_cfg = manual_params.get(feature, {})
    raw_range = feature_cfg.get("range", [default_low, default_high])

    if not isinstance(raw_range, list) or len(raw_range) != 2:
        return float(default_low), float(default_high)

    return float(raw_range[0]), float(raw_range[1])


def render_home_tab(
    dataframe: Optional[pd.DataFrame],
    errors: List[str],
    mf_payload: Optional[dict],
    anfis_model_error: Optional[str],
) -> None:
    st.subheader("Beranda & Exploratory Data Analysis (EDA)")
    st.markdown(
        """
        Aplikasi ini membandingkan **3 pendekatan AI** untuk prediksi risiko diabetes:
        **FIS Manual (Human Expert)**, **FIS + Genetic Algorithm (GA)**, dan **ANFIS (Neuro-Fuzzy)**.
        """
    )

    if errors:
        _render_errors(errors, "Sebagian aset wajib gagal dimuat:")

    if anfis_model_error:
        st.warning(
            "Model ANFIS PKL belum bisa dipakai secara langsung. Sistem menggunakan fallback surrogate agar prediksi tetap tersedia."
        )

    if mf_payload and "performance" in mf_payload:
        perf = mf_payload["performance"]
        summary_col, chart_col = st.columns([1.2, 1.0])
        with summary_col:
            c1, c2, c3 = st.columns(3)
            c1.metric("Akurasi FIS Manual", f"{float(perf.get('fis_manual_accuracy', 0.0)) * 100:.2f}%")
            c2.metric("Akurasi FIS + GA", f"{float(perf.get('fis_ga_accuracy', 0.0)) * 100:.2f}%")
            c3.metric("Akurasi ANFIS", f"{float(perf.get('anfis_accuracy', 0.0)) * 100:.2f}%")
            st.caption("Nilai akurasi diambil dari evaluasi model yang tersimpan pada file parameter.")
        with chart_col:
            plot_performance_bars(mf_payload)

    st.markdown("### Dataset dan EDA")
    if dataframe is None:
        st.warning("Dataset belum tersedia, halaman EDA tidak dapat ditampilkan.")
        return

    data_col, info_col = st.columns([1.25, 0.75])
    with data_col:
        st.markdown("#### Preview Dataset")
        st.dataframe(dataframe.head(15), use_container_width=True, height=340)

    with info_col:
        st.markdown("#### Ringkasan Data")
        st.write(f"Jumlah baris: {len(dataframe)}")
        st.write(f"Jumlah kolom: {len(dataframe.columns)}")
        if "Outcome" in dataframe.columns:
            positive_rate = float((dataframe["Outcome"] == 1).mean() * 100)
            st.write(f"Proporsi kelas positif: {positive_rate:.2f}%")
        st.caption("Visualisasi EDA menampilkan distribusi outcome dan korelasi fitur medis.")

    if "Outcome" not in dataframe.columns:
        st.warning("Kolom Outcome tidak ditemukan, visualisasi EDA tidak dapat ditampilkan.")
        return

    plot_eda_charts(dataframe)


def render_predict_tab(
    mf_payload: Optional[dict],
    anfis_model,
    errors: List[str],
    anfis_model_error: Optional[str],
) -> None:
    st.subheader("Simulator Prediksi Risiko Diabetes")
    st.markdown("Masukkan parameter pasien lalu klik tombol Prediksi Risiko.")

    if errors:
        _render_errors(errors, "Prediksi belum bisa dijalankan karena aset wajib belum lengkap:")
        return

    if not mf_payload:
        st.error("Parameter MF tidak tersedia.")
        return

    feature_order = mf_payload.get("features", DEFAULT_FEATURE_ORDER)
    default_map = {
        "Pregnancies": 2.0,
        "Glucose": 120.0,
        "BloodPressure": 70.0,
        "SkinThickness": 25.0,
        "Insulin": 120.0,
        "BMI": 28.0,
        "DiabetesPedigreeFunction": 0.47,
        "Age": 35.0,
    }

    with st.form("prediction_form"):
        columns = st.columns(3)
        input_map: Dict[str, float] = {}

        for idx, feature in enumerate(feature_order):
            col = columns[idx % 3]
            low, high = _feature_range(mf_payload, feature, 0.0, 100.0)
            default_val = float(default_map.get(feature, (low + high) / 2))

            if feature in ["Pregnancies", "Age", "SkinThickness", "Insulin"]:
                value = col.number_input(
                    feature,
                    min_value=int(np.floor(low)),
                    max_value=int(np.ceil(high)),
                    value=int(np.clip(default_val, low, high)),
                    step=1,
                )
            else:
                step = 0.01 if feature == "DiabetesPedigreeFunction" else 0.1
                value = col.number_input(
                    feature,
                    min_value=float(low),
                    max_value=float(high),
                    value=float(np.clip(default_val, low, high)),
                    step=float(step),
                )

            input_map[feature] = float(value)

        submitted = st.form_submit_button("Prediksi Risiko")

    if not submitted:
        return

    manual_score, manual_err = fuzzy_predict_risk(input_map, mf_payload, "manual")
    ga_score, ga_err = fuzzy_predict_risk(input_map, mf_payload, "ga")
    anfis_score, anfis_err = anfis_predict_risk(input_map, mf_payload, anfis_model)

    model_errors = [msg for msg in [manual_err, ga_err] if msg]
    if model_errors:
        _render_errors(model_errors, "Sebagian model gagal menghitung prediksi:")

    st.markdown("### Hasil Prediksi Tiga Metode")
    col_manual, col_ga, col_anfis = st.columns(3)

    if manual_score is not None:
        with col_manual:
            render_prediction_card("FIS Manual", manual_score)
    else:
        with col_manual:
            st.warning("FIS Manual belum menghasilkan skor.")

    if ga_score is not None:
        with col_ga:
            render_prediction_card("FIS + GA", ga_score)
    else:
        with col_ga:
            st.warning("FIS + GA belum menghasilkan skor.")

    if anfis_score is not None:
        with col_anfis:
            render_prediction_card("ANFIS", anfis_score)
    else:
        with col_anfis:
            st.warning("ANFIS belum menghasilkan skor.")

    if anfis_model_error:
        st.info("Model ANFIS PKL tidak tersedia, sistem memakai fallback surrogate ANFIS.")
    if anfis_err:
        st.info(anfis_err)

    available_scores = [score for score in [manual_score, ga_score, anfis_score] if score is not None]
    if available_scores:
        score_mean = float(np.mean(available_scores))
        st.markdown("---")
        st.write(
            f"Rata-rata risiko gabungan: **{score_mean * 100:.1f}%** "
            f"(kategori **{risk_label(score_mean)}**)."
        )


def render_shift_tab(mf_payload: Optional[dict], errors: List[str]) -> None:
    st.subheader("Perbandingan Kurva Membership Function")
    st.markdown("Tab ini menampilkan beberapa perbandingan kurva MF tiap model dan interpretasinya.")

    if errors:
        _render_errors(errors, "Visualisasi kurva tidak dapat ditampilkan karena aset wajib gagal dimuat:")
        return

    if not mf_payload:
        st.error("Parameter MF tidak tersedia.")
        return

    plot_shift_analysis(mf_payload)

    st.markdown(
        """
        ### Interpretasi
        - **FIS Manual** memakai kurva berbasis desain pakar, sehingga batas antar kategori tetap jelas dan mudah dijelaskan.
        - **FIS + GA** mempertahankan bentuk fuzzy klasik, tetapi titik kurva bergeser agar lebih sesuai dengan pola data.
        - **ANFIS** menggunakan Gaussian MF yang dipelajari dari data sehingga transisi antar kategori cenderung lebih halus.
        - Perbedaan bentuk MF ini memengaruhi sensitivitas model terhadap perubahan nilai fitur klinis.
        """
    )
