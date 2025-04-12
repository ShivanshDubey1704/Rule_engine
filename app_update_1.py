import streamlit as st
import pandas as pd
import requests
import subprocess
import threading
import time
import json
from io import BytesIO
import plotly.express as px

# Streamlit config
st.set_page_config("Smart Rule Engine", layout="wide")
st.title("🧠 Smart Rule Engine — Real-Time Anomaly Detection")

# Start Ollama Backend
@st.cache_resource
def start_backend():
    def run_backend():
        subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    thread = threading.Thread(target=run_backend)
    thread.start()
    time.sleep(3)
    return thread

# Load LLM Model
@st.cache_resource
def load_model(model_name='llama3'):
    def pull():
        subprocess.run(['ollama', 'pull', model_name], check=True)
    thread = threading.Thread(target=pull)
    thread.start()
    return thread

# Prompt Generation
def create_prompt(df: pd.DataFrame, max_chars=5000):
    sample = df.to_markdown(index=False)
    sample = sample[:max_chars]
    return f"""
You are a smart anomaly detector and rule engine.

Analyze the following dataset and:
1. Detect anomalies in the data.
2. Categorize the anomalies dynamically.
3. Define human-readable rules for each anomaly category.
4. Classify types of fraud or suspicious behavior.
5. Suggest industry type based on data patterns.
6. Output a JSON with:
   - "anomalies": [ list of anomaly summaries ],
   - "categories": [ {{ "name": ..., "description": ..., "rule": ..., "fraud_type": ... }} ],
   - "kpis": {{
       "anomaly_count": ..., 
       "total_rows": ..., 
       "columns_with_anomalies": [ list of column names ],
       "top_categories": [ {{ "name": ..., "count": ... }} ],
       "suggested_industry": ...
   }},
   - "possible_rules": [ list of potential fraud rules that can be written based on the dataset columns ]

Dataset:
{sample}
"""

# Stream LLM with Callback
def stream_llm_json_live(prompt, model_name='llama3', callback=None):
    url = "http://localhost:11434/api/generate"
    payload = {"model": model_name, "prompt": prompt, "stream": True}
    full_response = ""

    def fetch():
        nonlocal full_response
        with requests.post(url, json=payload, stream=True) as response:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    text = data.get("response", "")
                    full_response += text
                    if callback:
                        callback(text)

    thread = threading.Thread(target=fetch)
    thread.start()
    return thread, lambda: full_response

# Extract JSON Block
def extract_json(text):
    try:
        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON object found.")
        for end in range(len(text) - 1, start, -1):
            try:
                json_str = text[start:end + 1]
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
        raise ValueError("No complete JSON object found.")
    except Exception as e:
        st.error(f"❌ Error parsing LLM output: {e}")
        return {}

# Summary Dashboard
def show_summary_dashboard(df):
    st.subheader("📊 Summary Overview")
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Missing Values", df.isna().sum().sum())

    with st.expander("🔍 Column-level Analysis"):
        mv = df.isnull().sum()
        mv = mv[mv > 0]
        if not mv.empty:
            st.plotly_chart(px.bar(mv, orientation='h', labels={'index': 'Column', 'value': 'Missing Count'},
                                  title="Missing Values by Column"), use_container_width=True)

        st.markdown("### Numeric Column Distributions")
        num_df = df.select_dtypes(include='number')
        for col in num_df.columns[:3]:
            st.plotly_chart(px.histogram(df, x=col, title=f"Distribution of {col}"), use_container_width=True)

        st.markdown("### Top Unique Values (First 5 Columns)")
        cat_cols = df.select_dtypes(include='object').iloc[:, :5]
        for col in cat_cols.columns:
            top_vals = cat_cols[col].value_counts().head(5)
            st.plotly_chart(px.bar(top_vals, labels={'index': col, 'value': 'Count'},
                                   title=f"{col} Top Values"), use_container_width=True)

# Anomaly Dashboard
def show_anomaly_dashboard(results):
    st.subheader("📌 Anomaly Insights")
    with st.container(border=True):
        kpis = results.get("kpis", {})
        col1, col2, col3 = st.columns(3)
        col1.metric("🔍 Detected Anomalies", kpis.get("anomaly_count", 0))
        col2.metric("📄 Total Records", kpis.get("total_rows", 0))
        col3.metric("📉 Anomaly Rate", f"{(kpis.get('anomaly_count', 0) / kpis.get('total_rows', 1)) * 100:.2f}%")

    if kpis.get("suggested_industry"):
        st.success(f"🏭 Suggested Industry Type: {kpis['suggested_industry']}")

    if kpis.get("columns_with_anomalies"):
        col_df = pd.DataFrame(kpis["columns_with_anomalies"], columns=["Column"])
        st.plotly_chart(px.histogram(col_df, y="Column", title="Affected Data Columns"), use_container_width=True)

    if kpis.get("top_categories"):
        st.markdown("### 📊 Anomalies by Category")
        cat_df = pd.DataFrame(kpis["top_categories"])
        st.plotly_chart(px.bar(cat_df, x="count", y="name", orientation="h",
                               title="Top Anomaly Categories"), use_container_width=True)

    st.markdown("### 📘 Rule Categories")
    for cat in results.get("categories", []):
        with st.expander(f"🔹 {cat['name']}"):
            st.write(f"**Rule**: {cat['rule']}")
            st.write(f"**Description**: {cat['description']}")
            if cat.get("fraud_type"):
                st.warning(f"⚠️ Fraud Type: {cat['fraud_type']}")

    if results.get("anomalies"):
        st.markdown("### 📋 Anomaly Summary")
        for a in results.get("anomalies", []):
            st.markdown(f"- {a}")

        if st.button("💾 Download Anomaly Report"):
            anomaly_data = pd.DataFrame({"Anomaly": results.get("anomalies", [])})
            buffer = BytesIO()
            anomaly_data.to_excel(buffer, index=False)
            buffer.seek(0)
            st.download_button("Download Excel", data=buffer, file_name="anomaly_report.xlsx")

# Possible Rules
def show_possible_rules(results):
    st.subheader("🛡️ Suggested Detection Rules")
    rules = results.get("possible_rules", [])
    if rules:
        with st.container(border=True):
            for i, r in enumerate(rules, 1):
                st.markdown(f"**{i}.** {r}")
    else:
        st.info("No rules found.")

# UI Tabs
about_tab, main_tab, dashboard_tab, rules_tab = st.tabs([
    "\U0001F4D8 About Tool", "\U0001F4C5 Upload & Analyze", "\U0001F4CA Business Dashboard", "\U0001F6E1\uFE0F Suggested Rules"])

with about_tab:
    st.markdown("""
    ## 🧠 Smart Rule Engine — Overview

    The **Smart Rule Engine** is an AI-powered anomaly detection tool that uses Large Language Models (LLMs) to detect irregularities and suspicious patterns in datasets (such as Excel files). It:
    - Identifies anomalies across structured data,
    - Categorizes types of fraud or errors,
    - Automatically defines dynamic, human-readable rules,
    - Suggests the most likely industry based on dataset behavior.

    ---

    ## 💼 Business Use Case

    Modern organizations handle large volumes of transaction, audit, and operational data. Detecting anomalies or fraudulent activity manually is slow and error-prone.

    **Smart Rule Engine automates this by**:
    - Scanning entire datasets for inconsistencies,
    - Generating rule-based and statistical anomaly insights,
    - Creating compliance-ready summaries and recommendations,
    - Enabling quick detection of fraud, policy violations, and risky behavior.

    ---

    ## 🏭 Target Industries

    | Industry | Application |
    |----------|-------------|
    | **Banking & Fintech** | Fraudulent transactions, AML, unusual account activity |
    | **Retail & E-commerce** | Pricing mistakes, refund abuse, inventory mismatches |
    | **Healthcare** | Billing anomalies, duplicate procedures, claim fraud |
    | **Manufacturing** | Quality assurance issues, cost overruns |
    | **Travel & Expense (T&E)** | Fake bills, duplicate claims, out-of-policy expenses |
    | **Government & NGOs** | Grant misuse, procurement fraud, fund diversion |

    This tool helps **reduce financial risk**, **streamline audits**, and **ensure regulatory compliance** with AI-powered clarity.
    """, unsafe_allow_html=True)

with main_tab:
    uploaded_file = st.file_uploader("📂 Upload Excel File", type=["xlsx", "xls"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.success("✅ File uploaded successfully!")
        show_summary_dashboard(df)

        if st.button("🚀 Run Smart Rule Engine"):
            start_backend()
            load_model("llama3")

            prompt = create_prompt(df)
            output_text = []
            status = st.empty()
            progress = st.progress(0)

            def update_ui(chunk):
                output_text.append(chunk)

            status.info("🔍 Analyzing anomalies...")
            thread, get_result = stream_llm_json_live(prompt, callback=update_ui)

            progress_val = 0
            while thread.is_alive():
                time.sleep(0.3)
                progress_val = min(progress_val + 1, 100)
                progress.progress(progress_val)

            status.success("✅ Analysis complete!")
            progress.empty()

            results = extract_json(get_result())
            st.session_state.results = results

with dashboard_tab:
    if "results" in st.session_state:
        show_anomaly_dashboard(st.session_state.results)
    else:
        st.info("Run the analysis in the first tab.")

with rules_tab:
    if "results" in st.session_state:
        show_possible_rules(st.session_state.results)
    else:
        st.info("Run the analysis to see fraud rules.")
