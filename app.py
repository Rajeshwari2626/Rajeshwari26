import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Set up the Streamlit app
st.set_page_config(page_title="Social Media Sentiment Analyzer", layout="centered")
st.title("📊 Social Media Sentiment Analyzer")
st.markdown("Analyze the sentiment of your social media posts using VADER.")

# Text input for user
user_input = st.text_area("Enter your social media post here:")

# Analyze sentiment when the user clicks the button
if st.button("Analyze Sentiment"):
    if user_input:
        sentiment_score = analyzer.polarity_scores(user_input)
        compound = sentiment_score['compound']
        if compound >= 0.05:
            sentiment = 'Positive'
        elif compound <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        # Display the results
        st.subheader("Analysis Results:")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Sentiment Score:** {compound}")

        # Visualize the sentiment scores
        st.subheader("Sentiment Score Breakdown:")
        scores_df = pd.DataFrame([sentiment_score])
        st.bar_chart(scores_df)
    else:
        st.warning("Please enter text to analyze.")

