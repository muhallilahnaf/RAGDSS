import streamlit as st
import pandas as pd
import json
import io
import os
from dotenv import load_dotenv

load_dotenv()

from ragdss import initialize, run_query, NoRowsError

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAGDSS",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lato:ital,wght@0,100;0,300;0,400;0,700;0,900;1,100;1,300;1,400;1,700;1,900&display=swap');

html, body, [class*="css"] {
    font-family: 'Lato', sans-serif;
}

/* Background */
.stApp {
    background-color: #0f1117;
    color: #e2e8f0;
}

/* Main container */
section.main > div {
    padding-top: 2rem;
}

/* Header */
.dss-header {
    border-bottom: 1px solid #2d3748;
    padding-bottom: 1.2rem;
    margin-bottom: 2rem;
}
.dss-title {
    font-family: 'Lato', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #63b3ed;
    letter-spacing: -0.02em;
    margin: 0;
}
.dss-subtitle {
    font-size: 0.85rem;
    color: #718096;
    margin-top: 0.3rem;
    font-weight: 300;
}

/* Input area */
.stTextArea textarea {
    background-color: #1a202c !important;
    border: 1px solid #2d3748 !important;
    border-radius: 6px !important;
    color: #e2e8f0 !important;
    font-family: 'Lato', sans-serif !important;
    font-size: 0.95rem !important;
}
.stTextArea textarea:focus {
    border-color: #63b3ed !important;
    box-shadow: 0 0 0 1px #63b3ed !important;
}

/* Buttons */
.stButton > button {
    background-color: #2b6cb0 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 5px !important;
    font-family: 'Lato', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.4rem !important;
    transition: background-color 0.2s ease !important;
}
.stButton > button:hover {
    background-color: #3182ce !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background-color: #1a202c !important;
    border: 1px solid #2d3748 !important;
    border-radius: 6px !important;
    font-family: 'Lato', monospace !important;
    font-size: 0.82rem !important;
    color: #a0aec0 !important;
    font-weight: 600 !important;
}
.streamlit-expanderContent {
    background-color: #141820 !important;
    border: 1px solid #2d3748 !important;
    border-top: none !important;
    border-radius: 0 0 6px 6px !important;
}
            
.stExpander details summary span {
    background-color: #black !important;
}

/* Code blocks */
.stCodeBlock {
    border-radius: 6px !important;
}

/* Answer box */
.answer-box {
    background: linear-gradient(135deg, #1a2744 0%, #1a202c 100%);
    border: 1px solid #2b6cb0;
    border-left: 3px solid #63b3ed;
    border-radius: 8px;
    padding: 1.4rem 1.6rem;
    font-size: 0.95rem;
    line-height: 1.7;
    color: #e2e8f0;
    margin-top: 0.5rem;
}

/* Error box */
.error-box {
    background-color: #1a1020;
    border: 1px solid #744210;
    border-left: 3px solid #ed8936;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    color: #fbd38d;
    font-size: 0.9rem;
    margin-bottom: 1rem;
}

/* Section label */
.section-label {
    font-family: 'Lato', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}

/* History item */
.history-item {
    background-color: #1a202c;
    border: 1px solid #2d3748;
    border-radius: 6px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.82rem;
    color: #a0aec0;
    cursor: pointer;
    transition: border-color 0.15s;
}
.history-item:hover {
    border-color: #4a5568;
}
.history-q {
    color: #63b3ed;
    font-weight: 600;
    margin-bottom: 0.3rem;
    font-family: 'Lato', sans-serif;
}
.history-meta {
    font-family: 'Lato', monospace;
    font-size: 0.72rem;
    color: #4a5568;
}

/* Metric chips */
.metric-chip {
    display: inline-block;
    background-color: #1a202c;
    border: 1px solid #2d3748;
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-family: 'Lato', monospace;
    font-size: 0.75rem;
    color: #718096;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
}
.metric-chip span {
    color: #63b3ed;
    font-weight: 600;
}

/* Divider */
hr {
    border-color: #2d3748 !important;
    margin: 1.5rem 0 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0d1117 !important;
    border-right: 1px solid #1e2533 !important;
}
[data-testid="stSidebar"] .stMarkdown {
    color: #a0aec0;
}

/* Dataframe */
.stDataFrame {
    border-radius: 6px;
    overflow: hidden;
}

/* Review text cards */
.review-card {
    background-color: #1a202c;
    border: 1px solid #2d3748;
    border-radius: 6px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
    font-size: 0.88rem;
    line-height: 1.6;
    color: #cbd5e0;
}
.review-num {
    font-family: 'Lato', monospace;
    font-size: 0.68rem;
    color: #4a5568;
    margin-bottom: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)


# ── Session State Init ────────────────────────────────────────────────────────

def init_session():
    if "dss_state" not in st.session_state:
        with st.spinner("Initializing system..."):
            try:
                st.session_state.dss_state = initialize()
                st.session_state.init_error = None
            except Exception as e:
                st.session_state.dss_state = None
                st.session_state.init_error = str(e)

    if "history" not in st.session_state:
        st.session_state.history = []  # list of result dicts

    if "current_result" not in st.session_state:
        st.session_state.current_result = None

    if "no_rows_error" not in st.session_state:
        st.session_state.no_rows_error = None

    if "question_input" not in st.session_state:
        st.session_state.question_input = ""


init_session()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='font-family: Lato, monospace; font-size: 0.7rem;
                color: #4a5568; text-transform: uppercase; letter-spacing: 0.1em;
                margin-bottom: 1rem;'>
        Query History
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown(
            "<div style='font-size:0.8rem; color:#4a5568; font-style:italic;'>"
            "No queries yet.</div>",
            unsafe_allow_html=True,
        )
    else:
        for i, item in enumerate(reversed(st.session_state.history)):
            idx = len(st.session_state.history) - 1 - i
            with st.container():
                if st.button(
                    f"Q{idx+1}: {item['question'][:55]}{'...' if len(item['question'])>55 else ''}",
                    key=f"hist_{idx}",
                    use_container_width=True,
                ):
                    st.session_state.current_result = item
                    st.session_state.no_rows_error = None
                    st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family: Lato, monospace; font-size: 0.7rem;
                color: #4a5568; text-transform: uppercase; letter-spacing: 0.1em;
                margin-bottom: 0.8rem;'>
        About
    </div>
    <div style='font-size: 0.78rem; color: #4a5568; line-height: 1.6;'>
        RAG-based Decision Support System.<br><br>
        Ask business questions in plain English.
        The system filters products via SQL, retrieves
        semantically relevant reviews, and synthesizes
        an analyst-grade answer.
    </div>
    """, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="dss-header">
  <div class="dss-title">🔍 RAGDSS</div>
  <div class="dss-subtitle">
      Ask a business question — get answers grounded in customer reviews.
  </div>
</div>
""", unsafe_allow_html=True)


# ── Init Error ────────────────────────────────────────────────────────────────

if st.session_state.init_error:
    st.markdown(f"""
    <div class="error-box">
        <strong>Initialization failed:</strong><br>{st.session_state.init_error}<br><br>
        Make sure your <code>.env</code> file contains <code>OPENROUTER_API_KEY</code>
        and that <code>data/prices_clean.csv</code> and <code>data/reviews_clean.csv</code> exist.
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Question Input ────────────────────────────────────────────────────────────

col_input, col_btn = st.columns([5, 1])

with col_input:
    question = st.text_area(
        label="Business Question",
        placeholder='e.g. "Which Samsung phones under $500 have battery issues?"',
        height=80,
        label_visibility="collapsed",
        key="question_area",
        value=st.session_state.question_input,
    )

with col_btn:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    run_clicked = st.button("Run →", use_container_width=True)
    if st.session_state.history:
        clear_clicked = st.button("Clear", use_container_width=True)
        if clear_clicked:
            st.session_state.history = []
            st.session_state.current_result = None
            st.session_state.no_rows_error = None
            st.session_state.question_input = ""
            st.rerun()


# ── No-Rows Retry Warning ─────────────────────────────────────────────────────

if st.session_state.no_rows_error:
    err = st.session_state.no_rows_error
    st.markdown(f"""
    <div class="error-box">
        <strong>⚠ No products matched your query.</strong><br><br>
        The SQL below returned no results — this is usually due to an unrecognized
        product name or overly strict filters. Please rephrase your question using
        the exact product names or brands listed below.<br><br>
        <strong>SQL attempted:</strong><br>
        <code>{err['validated_sql']}</code>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Available Brands & Product Names"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Brands**")
            st.dataframe(
                pd.DataFrame(err["brands"], columns=["Brand"]),
                use_container_width=True,
                hide_index=True,
            )
        with c2:
            st.markdown("**Product Names**")
            st.dataframe(
                pd.DataFrame(err["prod_names"], columns=["Name"]),
                use_container_width=True,
                hide_index=True,
            )


# ── Run Pipeline ──────────────────────────────────────────────────────────────

if run_clicked and question.strip():
    st.session_state.no_rows_error = None

    with st.spinner("Running pipeline..."):
        try:
            result = run_query(question.strip(), st.session_state.dss_state)
            st.session_state.current_result = result
            st.session_state.history.append(result)
            st.session_state.question_input = ""
            st.rerun()

        except NoRowsError as e:
            st.session_state.no_rows_error = {
                "validated_sql": e.validated_sql,
                "filter_json": e.filter_json,
                "brands": e.brands,
                "prod_names": e.prod_names,
            }
            st.session_state.current_result = None
            st.rerun()

        except Exception as e:
            st.error(f"An error occurred: {e}")

elif run_clicked and not question.strip():
    st.warning("Please enter a question before running.")


# ── Results Display ───────────────────────────────────────────────────────────

result = st.session_state.current_result

if result:
    st.markdown("<hr>", unsafe_allow_html=True)

    # Metrics row
    n_products = len(result["rows"])
    n_reviews = len(result["df_filtered"])
    n_retrieved = len(result["retrieved_reviews"])

    st.markdown(f"""
    <div style='margin-bottom:1.2rem;'>
        <span class='metric-chip'>Products matched: <span>{n_products}</span></span>
        <span class='metric-chip'>Reviews found: <span>{n_reviews}</span></span>
        <span class='metric-chip'>Reviews used: <span>{n_retrieved}</span></span>
    </div>
    """, unsafe_allow_html=True)

    # ── Expander 1: Filter JSON ───────────────────────────────────────────────
    with st.expander("🗂  Filter JSON — extracted structured filters"):
        st.code(json.dumps(result["filter_json"], indent=2), language="json")

    # ── Expander 2: SQL ───────────────────────────────────────────────────────
    with st.expander("🛢  SQL Query — validated & executed"):
        st.code(result["validated_sql"], language="sql")

    # ── Expander 3: Filtered Reviews Table + Download ─────────────────────────
    with st.expander(f"📋  Filtered Reviews — {n_reviews} review(s) for matched product(s)"):
        df_display = result["df_filtered"].reset_index(drop=True)
        st.dataframe(df_display, use_container_width=True, height=280)

        csv_bytes = df_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇  Download as CSV",
            data=csv_bytes,
            file_name="filtered_reviews.csv",
            mime="text/csv",
        )

    # ── Expander 4: Retrieved Review Texts ───────────────────────────────────
    with st.expander(f"🔎  Top Retrieved Reviews — {n_retrieved} semantically matched"):
        st.markdown(
            f"<div class='section-label'>Semantic query used: "
            f"\"{result['semantic_query']}\"</div>",
            unsafe_allow_html=True,
        )
        for i, review_text in enumerate(result["retrieved_reviews"], 1):
            st.markdown(f"""
            <div class="review-card">
                <div class="review-num">Review {i} of {n_retrieved}</div>
                {review_text}
            </div>
            """, unsafe_allow_html=True)

    # ── Final Answer ──────────────────────────────────────────────────────────
    st.markdown("<div class='section-label' style='margin-top:1.2rem;'>Final Answer</div>",
                unsafe_allow_html=True)
    st.markdown(f"<div class='answer-box'>{result['answer']}</div>",
                unsafe_allow_html=True)

elif not st.session_state.no_rows_error:
    st.markdown("""
    <div style='text-align:center; padding: 4rem 0; color: #2d3748;'>
        <div style='font-size:2.5rem; margin-bottom:1rem;'>🔍</div>
        <div style='font-family: Lato, monospace; font-size:0.85rem;'>
            Enter a business question above to get started.
        </div>
    </div>
    """, unsafe_allow_html=True)
