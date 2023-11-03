from flask import Flask, render_template, request

import os
import googleapiclient.discovery
import googleapiclient.errors
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def get_youtube_comments_and_analyze_sentiment(video_id):
    api_service_name = "youtube"
    api_version = "v3"
    api_key = "AIzaSyCHxB3D52xkPHy8PvIDVw1HMoHWey68gUw"  # Replace with your actual API key

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=api_key
    )

    try:
        results = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100
        ).execute()

        comments = []
        for item in results["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        if comments:
            comments_and_sentiments = analyze_sentiment(comments)
            return comments_and_sentiments
        else:
            return []

    except googleapiclient.errors.HttpError as e:
        print("Error accessing the YouTube API:", e)
        return []

def analyze_sentiment(comments):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(comments)

    sequences = tokenizer.texts_to_sequences(comments)

    padded_sequences = pad_sequences(sequences)

    # Analyze sentiment using VADER
    sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = []

    for comment in comments:
        sentiment = sentiment_analyzer.polarity_scores(comment)
        # Determine if the comment is positive or negative based on the compound score
        if sentiment['compound'] >= 0.05:
            sentiment_label = 'Positive'
        elif sentiment['compound'] <= -0.05:
            sentiment_label = 'Negative'
        else:
            sentiment_label = 'Neutral'
        sentiment_scores.append((sentiment_label, sentiment))

    comments_and_sentiments = list(zip(comments, sentiment_scores))

    return comments_and_sentiments

video_id = "Po3jStA673E"
comments_and_sentiments = get_youtube_comments_and_analyze_sentiment(video_id)

for comment, (sentiment_label, sentiment) in comments_and_sentiments:
    print("Comment:", comment)
    print("Sentiment:", sentiment)
    print("Sentiment Label:", sentiment_label)
    print("\n")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    video_id = request.form['video_id']
    comments_and_sentiments = get_youtube_comments_and_analyze_sentiment(video_id)
    return render_template('results.html', comments_and_sentiments=comments_and_sentiments)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
