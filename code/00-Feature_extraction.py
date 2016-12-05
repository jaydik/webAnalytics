import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.stem import PorterStemmer

# Import the data
data = pd.read_csv('../data/raw/uci-news-aggregator.csv')
category = data['CATEGORY']
data = data.drop('CATEGORY', axis=1)

# Split test/train
X_train, X_test, y_train, y_test = train_test_split(data, category, test_size=0.25, random_state=8675309)

# Declare a Stemmer
stemmer = PorterStemmer()

# Add columns for tokenized and stemmed data
X_train['TOKENIZED'] = X_train['TITLE'].apply(word_tokenize)
X_train['STEMMED'] = X_train['TOKENIZED'].apply(lambda x: ' '.join([stemmer.stem(y) for y in x]))
X_test['TOKENIZED'] = X_test['TITLE'].apply(word_tokenize)
X_test['STEMMED'] = X_test['TOKENIZED'].apply(lambda x: ' '.join([stemmer.stem(y) for y in x]))

# Declare and fit the count vectorizer, transform the dataset
vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(X_train['STEMMED'])
X_test = vectorizer.transform(X_test['STEMMED'])

# Dump the datasets to disk
joblib.dump(X_test, '../data/processed/X_test.pkl')
joblib.dump(X_train, '../data/processed/X_train.pkl')
joblib.dump(y_test, '../data/processed/y_test.pkl')
joblib.dump(y_train, '../data/processed/y_train.pkl')
