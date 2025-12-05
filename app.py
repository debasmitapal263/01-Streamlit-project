import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from streamlit_echarts import st_echarts

st.title("Data Visualization Application")
st.header("Upload your data file here (CSV): ")

file = st.file_uploader("Upload your file here", type=["csv"])

if file is not None:
    data = pd.read_csv(file)
    data=data.fillna(0)
    st.subheader("Preview of Dataset")
    st.dataframe(data.head())

    # Select annotation column
    annotation = st.selectbox("Select a column for distribution", data.columns)

    # Select irrelevant columns (DO NOT remove annotation)
    st.subheader("Select irrelevant columns (except annotation)")
    irrelevant = st.multiselect(
        "Choose columns to remove:",
        [col for col in data.columns if col != annotation]
    )

    # Drop irrelevant columns but keep annotation
    if irrelevant:
        data = data.drop(columns=irrelevant)
    st.success("Data processed successfully!")

    st.subheader("Choose visualizations to display:")
    show_pie = st.checkbox("Data distribution (Pie Chart)")
    show_hist = st.checkbox("Statistical feature distribution (Histogram/KDE)")
    show_corr = st.checkbox("Correlation (scatterplot between two features)")
    show_heatmap = st.checkbox("Heatmap of selected features")

    # ------------------------------ PIE CHART ------------------------------
    if show_pie:
        st.markdown("### 📊 Class Distribution (Pie Chart)")
        dist = data[annotation].value_counts()

        chart_data = [
            {"name": str(category), "value": int(count)}
            for category, count in dist.items()
        ]

        option = {
            "tooltip": {"trigger": "item"},
            "series": [
                {
                    "type": "pie",
                    "radius": "60%",
                    "data": chart_data,
                    "label": {"formatter": "{b}: {d}%"},
                }
            ],
        }

        st_echarts(options=option, height="450px")

    # ------------------------------ HISTOGRAM ------------------------------
    if show_hist:
        st.markdown("### 📈 Statistical Feature Distribution")
        numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns

        if len(numeric_cols) == 0:
            st.error("No numeric features found!")
        else:
            feature = st.selectbox("Select a numeric feature:", numeric_cols)

            fig, ax = plt.subplots()
            sns.histplot(data=data, x=feature, hue=annotation, kde=True, palette="Set2", ax=ax)
            plt.title(f"Histogram & KDE of {feature} by {annotation}")
            st.pyplot(fig)

    # ------------------------------ CORRELATION SCATTER ------------------------------
    if show_corr:
        st.markdown("### 🔗 Correlation Between Two Features")
        numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns

        if len(numeric_cols) < 2:
            st.error("Need at least two numeric features!")
        else:
            f1 = st.selectbox("Select Feature 1:", numeric_cols)
            f2 = st.selectbox("Select Feature 2:", numeric_cols)

            fig, ax = plt.subplots()
            sns.scatterplot(x=data[f1], y=data[f2], hue=data[annotation], palette="Set2", ax=ax)
            plt.title(f"Scatterplot: {f1} vs {f2} by {annotation}")
            st.pyplot(fig)

    # ------------------------------ HEATMAP ------------------------------
    if show_heatmap:
        st.markdown("### 🔥 Heatmap of Selected Features")
        numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns

        selected = st.multiselect(
            "Choose numeric features for heatmap:", 
            numeric_cols, 
        )

        if len(selected) < 2:
            st.info("Select at least two numeric features.")
        else:
            # Create correlation for each class separately and plot as a multi-heatmap if desired
            fig, ax = plt.subplots(figsize=(6, 4))
            corr = data[selected].corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            plt.title("Correlation Heatmap (All Classes Combined)")
            st.pyplot(fig)