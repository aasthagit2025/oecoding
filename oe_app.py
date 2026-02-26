import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import text
from sklearn.cluster import KMeans

# -----------------------------
# STREAMLIT CONFIG
# -----------------------------
st.set_page_config(layout="wide")
st.title("AI Open-End Coding Tool (Stable Cloud Version)")

STOPWORDS = text.ENGLISH_STOP_WORDS

# -----------------------------
# CLEAN TEXT
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
# PROCESSING FUNCTION
# -----------------------------
@st.cache_data(show_spinner=False)
def process_data(df, text_column, n_clusters):

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

    # KMEANS CLUSTERING (STABLE)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    df["Cluster_ID"] = clusters

    # CONFIDENCE SCORING
    centroids = kmeans.cluster_centers_

    similarity = cosine_similarity(X, centroids)
    max_similarity = similarity.max(axis=1)

    df["Confidence_%"] = np.round(max_similarity * 100, 2)

    df["Sentiment"] = df[text_column].apply(simple_sentiment)

    return df, vectorizer

# -----------------------------
# KEYWORDS
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
    st.write("Available Columns:", df.columns)

    text_column = st.selectbox("Select Open-End Column", df.columns)

    # Choose number of clusters
    suggested_clusters = min(8, max(3, len(df)//100))
    n_clusters = st.slider("Select Number of Themes", 2, 15, suggested_clusters)

    if st.button("Run AI Coding"):

        with st.spinner("Processing..."):

            processed_df, vectorizer = process_data(df, text_column, n_clusters)
            keywords = extract_keywords(processed_df, vectorizer)

            st.session_state.processed_df = processed_df
            st.session_state.vectorizer = vectorizer
            st.session_state.keywords = keywords
            st.session_state.coding_done = True

    if st.session_state.coding_done:

        processed_df = st.session_state.processed_df
        keywords = st.session_state.keywords

        st.success("Clustering Completed")

        st.subheader("Rename Themes")

        cluster_names = {}

        for cluster, words in keywords.items():
            suggested_label = " / ".join(words[:3])
            cluster_names[cluster] = st.text_input(
                f"Theme {cluster}",
                value=suggested_label,
                key=f"cluster_{cluster}"
            )

        if st.button("Freeze Theme Names"):

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