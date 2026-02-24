from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

def prepare_features(df):
    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df['Cleaned_Tweet']).toarray()
    
    le = LabelEncoder()
    y = le.fit_transform(df['Sentiment'])
    
    return X, y, tfidf, le

def transform(df, tfidf, le):
    X = tfidf.transform(df['Cleaned_Tweet']).toarray()
    y = le.transform(df['Sentiment'])
    return X, y