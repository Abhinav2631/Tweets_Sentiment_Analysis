# Tweets Sentiment Analysis

## Overview
This project performs sentiment analysis on tweets using Machine Learning techniques. The model classifies tweets into categories such as Positive, Negative, Neutral, and Irrelevant.

## Features
- Text preprocessing using NLTK
- Feature extraction using TF-IDF
- Visualization using Matplotlib and Seaborn
- Machine Learning models:
  - Decision Tree
  - Random Forest
- Evaluation using accuracy, precision, recall, and F1-score

## Dataset
- twitter_training.csv → Training dataset
- twitter_validation.csv → Validation dataset

## Project Structure
Tweets_Sentiment_Analysis/
├── data/
│   ├── twitter_training.csv
│   └── twitter_validation.csv
├── notebooks/
│   └── Analysis.ipynb
├── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── model.py
│   ├── evaluate.py
│   └── predict.py
├── README.md
└── requirements.txt

## Installation
1. Clone the repository
2. Install dependencies

pip install -r requirements.txt

## Usage
Open the notebook and run all cells:

notebooks/Analysis.ipynb

## Results
The Random Forest model performs better than Decision Tree with higher accuracy.

## Visualizations
- PCA Scatter Plot
- Tweet Length Distribution
- Sentiment Distribution
- Correlation Heatmap

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn

## Author
Balla Abhinav