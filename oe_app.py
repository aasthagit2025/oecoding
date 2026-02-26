import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import hdbscan

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

# -----------------------------
# STREAMLIT CONFIG
# -----------------------------
st.set_page_config(layout="wide")
st.title("AI Open-End Coding Tool (Production Version)")

# -----------------------------
# LOAD NLTK (Assumes pre-installed in environment)
# -----------------------------
STOPWORDS = set(stopwords.words("english"))
sia = SentimentIntensityAnalyzer()

# -----------------------------
# TEXT CLEANING
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    return " ".join(words)

# -----------------------------
# CORE PROCESSING FUNCTION (CACHED)
# -----------------------------
@st.cache_data(show_spinner=False)
def process_data(df, text_column):

    # Remove blanks
    df = df[df[text_column].notna()]
    df = df[df[text_column].str.strip() != ""]
    df = df.reset_index(drop=True)

    # Clean text
    df["clean_text"] = df[text_column].apply(clean_text)

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=4000, ngram_range=(1,2))
    X = vectorizer.fit_transform(df["clean_text"])

    # Dimensionality reduction
    svd = TruncatedSVD(n_components=100, random_state=42)
    X_reduced = svd.fit_transform(X)

    # Clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(15, int(len(df)*0.01)),  # auto scale
        metric="euclidean"
    )
    clusters = clusterer.fit_predict(X_reduced)
    df["Cluster_ID"] = clusters

    # -----------------------------
    # VECTORISED CONFIDENCE SCORING
    # -----------------------------
    unique_clusters = np.unique(clusters)
    cluster_centers = {}

    for cluster in unique_clusters:
        if cluster == -1:
            continue
        cluster_centers[cluster] = X_reduced[clusters == cluster].mean(axis=0)

    confidence_scores = np.zeros(len(df))

    for cluster, center in cluster_centers.items():
        cluster_mask = df["Cluster_ID"] == cluster
        similarity = cosine_similarity(
            X_reduced[cluster_mask], 
            center.reshape(1, -1)
        ).flatten()
        confidence_scores[cluster_mask] = similarity * 100

    df["Confidence_%"] = np.round(confidence_scores, 2)

    # -----------------------------
    # SENTIMENT (Vectorized Apply)
    # -----------------------------
    df["Sentiment"] = df[text_column].apply(
        lambda x: "Positive" if sia.polarity_scores(str(x))["compound"] >= 0.05
        else "Negative" if sia.polarity_scores(str(x))["compound"] <= -0.05
        else "Neutral"
    )

    return df, vectorizer


# -----------------------------
# CLUSTER KEYWORDS
# -----------------------------
def extract_keywords(df, vectorizer):

    feature_names = vectorizer.get_feature_names_out()
    cluster_keywords = {}

    for cluster in df["Cluster_ID"].unique():
        if cluster == -1:
            continue

        cluster_text = df[df["Cluster_ID"] == cluster]["clean_text"]
        cluster_vector = vectorizer.transform(cluster_text)
        mean_tfidf = cluster_vector.mean(axis=0)
        sorted_indices = mean_tfidf.A1.argsort()[::-1][:10]
        top_words = [feature_names[i] for i in sorted_indices]

        cluster_keywords[cluster] = top_words

    return cluster_keywords


# -----------------------------
# UI
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Available Columns:", df.columns)

    text_column = st.selectbox("Select Open-End Column", df.columns)

    if st.button("Run AI Coding"):

        with st.spinner("Processing 10K+ verbatims... please wait."):

            processed_df, vectorizer = process_data(df, text_column)
            keywords = extract_keywords(processed_df, vectorizer)

        st.success("Clustering Completed")

        # -----------------------------
        # RENAME CLUSTERS
        # -----------------------------
        st.subheader("Rename Clusters")

        cluster_names = {}
        for cluster, words in keywords.items():
            suggested_label = " / ".join(words[:3])
            new_name = st.text_input(
                f"Cluster {cluster}",
                value=suggested_label
            )
            cluster_names[cluster] = new_name

        if st.button("Freeze Cluster Names"):

            processed_df["Cluster_Name"] = processed_df["Cluster_ID"].map(cluster_names)
            processed_df["Cluster_Name"] = processed_df["Cluster_Name"].fillna("Noise/Unclassified")

            # -----------------------------
            # FREQUENCY SUMMARY
            # -----------------------------
            freq_summary = (
                processed_df.groupby("Cluster_Name")
                .size()
                .reset_index(name="Count")
                .sort_values(by="Count", ascending=False)
            )

            freq_summary["%"] = np.round(
                (freq_summary["Count"] / len(processed_df)) * 100, 2
            )

            st.subheader("Theme Frequency Summary")
            st.dataframe(freq_summary)

            # -----------------------------
            # EXPORT
            # -----------------------------
            st.download_button(
                label="Download Coded Dataset",
                data=processed_df.to_csv(index=False),
                file_name="coded_output.csv",
                mime="text/csv"
            )

            st.download_button(
                label="Download Frequency Summary",
                data=freq_summary.to_csv(index=False),
                file_name="theme_summary.csv",
                mime="text/csv"
            )

            st.success("Coding Completed & Ready for Download.")