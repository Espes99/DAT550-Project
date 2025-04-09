import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import re
import nltk
from nltk.stem import WordNetLemmatizer

def read_data(path):
    df_train = pd.read_csv(path)
    df_train = df_train.dropna()
    X = df_train["abstract"]
    y = df_train["label"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return X_train, X_val, y_train, y_val

def clean_text_with_lemmatization(text):
    text = text.lower()
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    tokens = nltk.word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization - reduce words to their base form
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    cleaned_text = ' '.join(tokens)
    return cleaned_text

def clean_text_withtout_lemmatization(text):
    text = text.lower()
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    tokens = nltk.word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    cleaned_text = ' '.join(tokens)

    return cleaned_text

def preprocess_text_series(text_series: pd.Series, lem: bool) -> pd.Series:
    if lem:
        return text_series.apply(clean_text_with_lemmatization)
    else:
        return text_series.apply(clean_text_withtout_lemmatization)
