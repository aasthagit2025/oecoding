import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(layout="wide")
st.title("AI Open-End Coding Tool (Semantic Version)")

# Load model once
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------------
# CLEAN TEXT
# -----------------------------
def clean_text(text_input):
    text_input = str(text_input).strip()
    text_input = re.sub(r"\s+", " ", text_input)
    return text_input

# -----------------------------
# PROCESS DATA
# -----------------------------
@st.cache_data(show_spinner=False)
def process_data(df, text_column, n_clusters):

    df = df[df[text_column].notna()]
    df = df[df[text_column].str.strip() != ""]
    df = df.reset_index(drop=True)

    df["clean_text"] = df[text_column].apply(clean_text)

    # Generate semantic embeddings
    embeddings = model.encode(
        df["clean_text"].tolist(),
        show_progress_bar=False
    )

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)

    df["Cluster_ID"] = clusters

    # Confidence score
    similarity = cosine_similarity(embeddings, kmeans.cluster_centers_)
    df["Confidence_%"] = np.round(similarity.max(axis=1) * 100, 2)

    return df, embeddings

# -----------------------------
# EXTRACT KEYWORDS
# -----------------------------
def extract_keywords(df, text_column):

    cluster_keywords = {}

    for cluster in df["Cluster_ID"].unique():

        cluster_text = df[df["Cluster_ID"] == cluster][text_column]

        combined = " ".join(cluster_text.tolist())
        words = combined.lower().split()

        freq = pd.Series(words).value_counts().head(5)
        cluster_keywords[cluster] = list(freq.index)

    return cluster_keywords

# -----------------------------
# SESSION STATE
# -----------------------------
if "coding_done" not in st.session_state:
    st.session_state.coding_done = False

# -----------------------------
# UI
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    st.write("Columns:", df.columns)

    text_column = st.selectbox("Select Open-End Column", df.columns)

    suggested_clusters = min(8, max(3, len(df)//100))
    n_clusters = st.slider("Number of Themes", 2, 15, suggested_clusters)

    if st.button("Run AI Coding"):

        with st.spinner("Generating semantic clusters..."):

            processed_df, embeddings = process_data(df, text_column, n_clusters)
            keywords = extract_keywords(processed_df, text_column)

            st.session_state.processed_df = processed_df
            st.session_state.keywords = keywords
            st.session_state.coding_done = True

    # -----------------------------
    # RENAME THEMES
    # -----------------------------
    if st.session_state.coding_done:

        processed_df = st.session_state.processed_df
        keywords = st.session_state.keywords

        st.success("Semantic Clustering Completed")

        st.subheader("Rename Themes")

        cluster_names = {}

        for cluster, words in keywords.items():
            suggested_label = " / ".join(words[:3])
            cluster_names[cluster] = st.text_input(
                f"Theme {cluster}",
                value=suggested_label,
                key=f"cluster_{cluster}"
            )

        if st.button("Freeze Themes"):

            processed_df["Cluster_Name"] = processed_df["Cluster_ID"].map(cluster_names)

            freq_summary = (
                processed_df.groupby("Cluster_Name")
                .size()
                .reset_index(name="Count")
                .sort_values(by="Count", ascending=False)
            )

            freq_summary["%"] = np.round(
                (freq_summary["Count"] / len(processed_df)) * 100, 2
            )

            # Store final outputs
            st.session_state.final_df = processed_df
            st.session_state.freq_summary = freq_summary
            st.session_state.frozen = True

    # -----------------------------
    # DOWNLOAD SECTION (SEPARATE)
    # -----------------------------
    if st.session_state.get("frozen", False):

        st.subheader("Theme Frequency Summary")
        st.dataframe(st.session_state.freq_summary)

        st.download_button(
            "Download Coded Dataset",
            st.session_state.final_df.to_csv(index=False),
            "coded_output.csv"
        )

        st.download_button(
            "Download Frequency Summary",
            st.session_state.freq_summary.to_csv(index=False),
            "theme_summary.csv"
        )

        st.success("Downloads Ready.")