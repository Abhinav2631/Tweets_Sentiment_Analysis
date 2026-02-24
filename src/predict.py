def predict_sample(model, tfidf, le, clean_text):
    sample_tweets = [
        "I love the new update, it's amazing!",
        "This app keeps crashing. Worst experience ever!",
        "Not sure how I feel about the new feature.",
        "Absolutely terrible customer service.",
        "Flight was delayed but the crew was nice."
    ]

    cleaned_samples = [clean_text(tweet) for tweet in sample_tweets]
    sample_features = tfidf.transform(cleaned_samples).toarray()

    predicted_labels = model.predict(sample_features)
    predicted_sentiments = le.inverse_transform(predicted_labels)

    for tweet, sentiment in zip(sample_tweets, predicted_sentiments):
        print(f"Tweet: {tweet}")
        print(f"Predicted Sentiment: {sentiment}\n")