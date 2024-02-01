#Natural Language Processing (NLP) for Sentiment Analysis
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

def sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    if sentiment['compound'] >= 0.05:
        return 'Positive'
    elif sentiment['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

text = 'I love this product! It is amazing!'
print(sentiment_analysis(text)) # Output: Positive

text = 'I hate this product! It is terrible!'
print(sentiment_analysis(text)) # Output: Negative

text = 'This product is okay.'
print(sentiment_analysis(text)) # Output: Neutral
