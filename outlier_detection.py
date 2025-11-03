import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.robust.scale import mad

st.set_page_config(page_title="Outlier Detection App", layout="wide")

st.title("🔍 Detect Outliers Using Statistical Methods")
st.write("Upload a CSV file and detect unusual (outlier) data points using different statistical methods.")

# --- File Upload ---
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

# --- Method Selection ---
method = st.selectbox(
    "Select Outlier Detection Method:",
    ["Z-Score", "IQR", "Modified Z-Score"]
)

threshold = st.number_input("Enter Threshold Value:", min_value=0.0, value=3.0)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Data:")
    st.dataframe(df.head())

    # Keep only numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        st.warning("⚠️ No numeric columns found in the file.")
    else:
        st.write("Numeric columns detected:", numeric_cols)
        col = st.selectbox("Select Column to Analyze", numeric_cols)
        data = df[col]

        # --- Outlier Detection Functions ---
        def detect_zscore(series, th):
            z = (series - series.mean()) / series.std(ddof=0)
            return np.abs(z) > th

        def detect_iqr(series, k=1.5):
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - k * iqr
            upper = q3 + k * iqr
            return (series < lower) | (series > upper)

        def detect_modified_zscore(series, th):
            med = series.median()
            mad_val = mad(series)
            if mad_val == 0:
                z = (series - med) / series.std(ddof=0)
                return np.abs(z) > th
            mod_z = 0.6745 * (series - med) / mad_val
            return np.abs(mod_z) > th

        # --- Apply selected method ---
        if st.button("🚀 Detect Outliers"):
            if method == "Z-Score":
                mask = detect_zscore(data, threshold)
            elif method == "IQR":
                mask = detect_iqr(data)
            else:
                mask = detect_modified_zscore(data, threshold)

            outliers = df[mask]
            st.write(f"### Total Outliers Detected: {mask.sum()}")

            st.write("#### Outlier Rows:")
            st.dataframe(outliers)

            # --- Plot ---
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.scatter(df.index, data, label="Data", s=10)
            ax.scatter(df.index[mask], data[mask], color="red", label="Outliers", s=30)
            ax.set_title(f"{col} - {method} Outliers")
            ax.legend()
            st.pyplot(fig)
