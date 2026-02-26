import streamlit as st
import pandas as pd
import numpy as np
import re
import hdbscan

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import text
from sklearn.cluster import KMeans

# -----------------------------
# STREAMLIT CONFIG
# -----------------------------
st.set_page_config(layout="wide")
st.title("AI Open-End Coding Tool (Stable Production Version)")

# -----------------------------
# STOPWORDS
# -----------------------------
STOPWORDS = text.ENGLISH_STOP_WORDS

# -----------------------------
# TEXT CLEANING
# -----------------------------
def clean_text(text_input):
    text_input = str(text_input).lower()
    text_input = re.sub(r"[^a-zA-Z\s]", "", text_input)
    words = text_input.split()
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    return " ".join(words)

# -----------------------------
# SIMPLE SENTIMENT
# -----------------------------
positive_words = {"good", "great", "excellent", "fast", "easy", "love", "best", "happy"}
negative_words = {"bad", "poor", "slow", "late", "worst", "hate", "problem", "issue"}

def simple_sentiment(text_input):
    words = set(str(text_input).lower().split())
    if len(words & positive_words) > len(words & negative_words):
        return "Positive"
    elif len(words & negative_words) > len(words & positive_words):
        return "Negative"
    else:
        return "Neutral"

# -----------------------------
# CORE PROCESSING
# -----------------------------
@st.cache_data(show_spinner=False)
def process_data(df, text_column):

    df = df[df[text_column].notna()]
    df = df[df[text_column].str.strip() != ""]
    df = df.reset_index(drop=True)

    df["clean_text"] = df[text_column].apply(clean_text)

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2
    )

    X = vectorizer.fit_transform(df["clean_text"])

    # -----------------------------
    # PRIMARY CLUSTERING (HDBSCAN)
    # -----------------------------
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=2,
        metric="cosine"
    )

    clusters = clusterer.fit_predict(X.toarray())

    # -----------------------------
    # FALLBACK IF ALL NOISE
    # -----------------------------
    if len(set(clusters)) <= 1:
        n_clusters = min(8, max(2, len(df)//20))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)

    df["Cluster_ID"] = clusters

    # -----------------------------
    # CONFIDENCE SCORING
    # -----------------------------
    confidence_scores = np.zeros(len(df))
    unique_clusters = np.unique(clusters)

    for cluster in unique_clusters:
        mask = df["Cluster_ID"] == cluster
        cluster_points = X[mask]

        centroid = cluster_points.mean(axis=0)
        similarity = cosine_similarity(
            cluster_points,
            centroid
        ).flatten()

        confidence_scores[mask] = similarity * 100

    df["Confidence_%"] = np.round(confidence_scores, 2)

    # -----------------------------
    # SENTIMENT
    # -----------------------------
    df["Sentiment"] = df[text_column].apply(simple_sentiment)

    return df, vectorizer

# -----------------------------
# KEYWORD EXTRACTION
# -----------------------------
def extract_keywords(df, vectorizer):

    feature_names = vectorizer.get_feature_names_out()
    cluster_keywords = {}

    for cluster in df["Cluster_ID"].unique():

        cluster_text = df[df["Cluster_ID"] == cluster]["clean_text"]
        cluster_vector = vectorizer.transform(cluster_text)

        mean_tfidf = cluster_vector.mean(axis=0)
        sorted_indices = mean_tfidf.A1.argsort()[::-1][:10]

        top_words = [feature_names[i] for i in sorted_indices]
        cluster_keywords[cluster] = top_words

    return cluster_keywords

# -----------------------------
# SESSION STATE INIT
# -----------------------------
if "coding_done" not in st.session_state:
    st.session_state.coding_done = False

# -----------------------------
# UI
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    st.write("Available Columns:", df.columns)

    text_column = st.selectbox("Select Open-End Column", df.columns)

    # RUN CODING
    if st.button("Run AI Coding"):

        with st.spinner("Processing..."):

            processed_df, vectorizer = process_data(df, text_column)
            keywords = extract_keywords(processed_df, vectorizer)

            st.session_state.processed_df = processed_df
            st.session_state.vectorizer = vectorizer
            st.session_state.keywords = keywords
            st.session_state.coding_done = True

    # RENAME + FREEZE
    if st.session_state.coding_done:

        processed_df = st.session_state.processed_df
        keywords = st.session_state.keywords

        st.success("Clustering Completed")

        st.subheader("Rename Clusters")

        cluster_names = {}

        for cluster, words in keywords.items():
            suggested_label = " / ".join(words[:3])
            cluster_names[cluster] = st.text_input(
                f"Cluster {cluster}",
                value=suggested_label,
                key=f"cluster_{cluster}"
            )

        if st.button("Freeze Cluster Names"):

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

            st.subheader("Theme Frequency Summary")
            st.dataframe(freq_summary)

            st.download_button(
                "Download Coded Dataset",
                processed_df.to_csv(index=False),
                "coded_output.csv"
            )

            st.download_button(
                "Download Frequency Summary",
                freq_summary.to_csv(index=False),
                "theme_summary.csv"
            )

            st.success("Coding Completed & Ready for Download.")