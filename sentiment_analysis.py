import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import nltk
####

# Download the VADER lexicon (required for SentimentIntensityAnalyzer)
nltk.download('vader_lexicon')

# Function to perform sentiment analysis
def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_score = sid.polarity_scores(text)
    sentiment = 'Positive' if sentiment_score['compound'] > 0 else 'Negative' if sentiment_score['compound'] < 0 else 'Neutral'
    return sentiment, sentiment_score['compound']

# Streamlit app
def main():
    st.title("Sentiment Analysis App")
    st.sidebar.header("User Input")

    # Get user input text
    user_input = st.sidebar.text_area("Enter Text (Max 10,000 characters):", "", max_chars=10000)

    # Perform sentiment analysis
    if user_input:
        sentiment, compound_score = analyze_sentiment(user_input)

        # Display sentiment result
        st.header("Sentiment Analysis Result:")
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Compound Score: {compound_score:.2f}")

        # Display graphical representation
        st.header("Graphical Representation:")
        fig, ax = plt.subplots()
        ax.bar(["Negative", "Neutral", "Positive"], [abs(compound_score), 1 - abs(compound_score), abs(compound_score)],
               color=['red', 'grey', 'green'])
        ax.set_ylabel("Score")
        ax.set_title("Sentiment Analysis")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
