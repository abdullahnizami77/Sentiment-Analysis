
# Sentiment Analysis with Naive Bayes

This repository contains Python code for performing sentiment analysis on movie reviews using Naive Bayes classification. The dataset consists of movie reviews labeled as positive or negative, and we train a Naive Bayes classifier to predict the sentiment of new reviews.

## Requirements

- Python 3.x
- NumPy
- pandas
- scikit-learn

## Usage

1. Clone or download this repository to your local machine.

2. Make sure you have the required dependencies installed. You can install them using pip:


3. Execute the `sentiment_analysis.py` script to perform sentiment analysis on the provided movie reviews dataset.

## About the Dataset

The dataset contains 50,000 movie reviews, evenly split between positive and negative sentiments. It is used to train and evaluate the Naive Bayes classifier for sentiment analysis.

## How the Code Works

1. Load and preprocess the movie reviews dataset.
2. Clean the text data by removing special characters, HTML tags, and converting text to lowercase.
3. Split the dataset into training and testing sets.
4. Extract features from the text data using the CountVectorizer, which converts text into numerical features.
5. Train the Naive Bayes classifier using the training data.
6. Evaluate the classifier's performance on the testing data.
7. Save the trained classifier and the CountVectorizer to disk for future use.

## Example

You can use the saved model and vectorizer to predict the sentiment of new movie reviews. Here's a quick example:

```python
import pickle

# Load the saved model and vectorizer
with open('nb_model.pkl', 'rb') as f:
    nb_model = pickle.load(f)

with open('cv1.pkl', 'rb') as f:
    cv1 = pickle.load(f)

# New movie reviews for prediction
test_reviews = [
    "I do not like this movie",
    "I would not recommend this movie",
    "I hate this movie",
    "I love this movie"
]

# Clean the new reviews
cleaned_reviews = [clean(review) for review in test_reviews]

# Convert the cleaned reviews into numerical features using the vectorizer
features = cv1.transform(cleaned_reviews)

# Make predictions
predictions = nb_model.predict(features)

# Print the predictions
for review, sentiment in zip(test_reviews, predictions):
    print(f"Review: {review}")
    print(f"Sentiment: {'Positive' if sentiment == 1 else 'Negative'}")
    print()
