import streamlit as st

from .config import APP_TITLE


def configure_page() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🩸",
        layout="wide",
        initial_sidebar_state="collapsed",
    )


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --primary: #2e499d;
            --primary-dark: #243b82;
            --bg: #ffffff;
            --text: #1f2937;
            --muted: #5b6475;
            --border: #d9e1ef;
        }

        html, body, [data-testid="stAppViewContainer"], .stApp {
            background-color: var(--bg);
            color: var(--text);
        }

        body {
            color: var(--text);
        }

        [data-testid="stHeader"] {
            background-color: rgba(255, 255, 255, 0.95);
            border-bottom: 1px solid var(--border);
        }

        [data-testid="stSidebar"] {
            display: none !important;
        }

        [data-testid="collapsedControl"] {
            display: none !important;
        }

        .stApp .stCaption,
        .stApp small {
            color: var(--muted) !important;
        }

        .stMarkdown,
        .stMarkdown p,
        .stMarkdown li,
        .stSubheader,
        .stHeader,
        h1, h2, h3, h4, h5, h6,
        label {
            color: var(--text) !important;
        }

        a {
            color: var(--primary);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 1.5rem;
            border-bottom: 1px solid var(--border);
            margin-bottom: 0.75rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: transparent !important;
            border: none !important;
            border-radius: 0 !important;
            padding: 0.45rem 0.1rem 0.65rem 0.1rem;
            color: #475569 !important;
            font-weight: 600;
            height: auto;
            margin-bottom: -1px;
        }

        .stTabs [aria-selected="true"] {
            color: var(--primary) !important;
            font-weight: 700;
            border-bottom: 3px solid var(--primary) !important;
        }

        .stTabs [data-baseweb="tab-highlight"] {
            display: none;
        }

        div[data-testid="stFormSubmitButton"] > button {
            background-color: var(--primary);
            color: #ffffff !important;
            border: 1px solid var(--primary);
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease;
        }

        div[data-testid="stFormSubmitButton"] > button,
        div[data-testid="stFormSubmitButton"] > button:hover,
        div[data-testid="stFormSubmitButton"] > button:focus {
            color: #ffffff !important;
        }

        div[data-testid="stFormSubmitButton"] > button:hover {
            background-color: var(--primary-dark);
            border-color: var(--primary-dark);
        }

        div[data-testid="stFormSubmitButton"] > button:focus {
            box-shadow: 0 0 0 0.15rem rgba(46, 73, 157, 0.25);
        }

        div[data-testid="stMetricLabel"] p {
            color: var(--muted) !important;
        }

        div[data-testid="stMetricValue"] {
            color: var(--text);
        }

        div[data-testid="stProgressBar"] div[role="progressbar"] {
            background-color: var(--primary);
        }

        [data-testid="stDataFrame"] * {
            color: var(--text) !important;
        }

        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea {
            color: var(--text) !important;
            background-color: #ffffff !important;
            border-color: #cbd5e1 !important;
        }

        .section-card {
            border: 1px solid #e9edf5;
            border-radius: 10px;
            padding: 0.75rem 1rem;
            background-color: #ffffff;
        }

        .risk-label {
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--primary) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
