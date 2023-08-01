import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import streamlit as st

# Function to preprocess the text
def preprocess_text(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_words = [word.lower() for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)

# Read sentences from CSV or plain text file
def read_sentences(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        # Replace 'your_column_name' with the actual column name in your CSV file.
        sentences = df['challenges'].tolist()  # Replace 'your_column_name'
    else:
        sentences = uploaded_file.getvalue().decode('utf-8').splitlines()
    return sentences

# Bag-of-words vectorization
def bow_vectorization(sentences):
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(sentences)
    return bow_matrix, vectorizer

# Latent Dirichlet Allocation (LDA)
def perform_lda(bow_matrix, num_topics):
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(bow_matrix)
    return lda

# Extract the most relevant words for each topic
def extract_topic_keywords(lda, vectorizer, num_words=5):
    topic_keywords = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[-num_words:][::-1]
        top_words = [vectorizer.get_feature_names()[i] for i in top_words_idx]
        topic_keywords.append(top_words)
    return topic_keywords


# Generate theme names based on topic keywords
def generate_theme_names(topic_keywords):
    theme_names = []
    for idx, keywords in enumerate(topic_keywords):
        theme_name = " ".join(keywords)
        theme_names.append(f"Theme {idx + 1}: {theme_name}")
    return theme_names

# Function to draw pie chart
def draw_pie_chart(theme_names):
    theme_counts = [len(sentences) for sentences in theme_names]
    plt.figure(figsize=(8, 8))
    plt.pie(theme_counts, labels=theme_names, autopct="%1.1f%%", startangle=140)
    plt.axis("equal")
    plt.title("Theme Distribution")
    # plt.show()  # Remove this line, as st.pyplot() will handle the rendering.
    st.pyplot(plt.gcf())

# Function to draw bar chart
def draw_bar_chart(theme_names):
    theme_counts = [len(sentences) for sentences in theme_names]
    plt.figure(figsize=(10, 5))
    plt.bar(theme_names, theme_counts)
    plt.xlabel("Themes")
    plt.ylabel("Frequency")
    plt.title("Theme Frequency")
    plt.xticks(rotation=45)
    # plt.show()  # Remove this line, as st.pyplot() will handle the rendering.
    st.pyplot(plt.gcf())

# Main function
def main():
    st.title("Text Analytics: Theme Analysis of College Challenges")

    nltk.download("punkt")
    nltk.download("stopwords")

    st.write("Please upload a CSV file containing sentences.")
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])

    if uploaded_file is not None:
        num_topics = 5  # Number of themes to generate (you can adjust this)

        sentences = read_sentences(uploaded_file)
        preprocessed_sentences = [preprocess_text(sent) for sent in sentences]

        bow_matrix, vectorizer = bow_vectorization(preprocessed_sentences)
        lda = perform_lda(bow_matrix, num_topics)
        topic_keywords = extract_topic_keywords(lda, vectorizer)
        theme_names = generate_theme_names(topic_keywords)

        st.header("Themes:")
        for theme_name, sentences in zip(theme_names, topic_keywords):
            st.subheader(theme_name)
            for keyword in sentences:
                st.write("-", keyword)

        st.header("Theme Distribution:")
        draw_pie_chart(theme_names)

        st.header("Theme Frequency:")
        draw_bar_chart(theme_names)

if __name__ == "__main__":
     st.set_option('deprecation.showPyplotGlobalUse', False)  # Disable the warning for the entire script
     main()
