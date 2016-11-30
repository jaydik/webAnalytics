import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.stem import PorterStemmer

# Import the data
data = pd.read_csv('../data/raw/uci-news-aggregator.csv')

# Declare a Stemmer
stemmer = PorterStemmer()

# Add columns for tokenized and stemmed data
data['TOKENIZED'] = data['TITLE'].apply(word_tokenize)
data['STEMMED'] = data['TOKENIZED'].apply(lambda x: ' '.join([stemmer.stem(y) for y in x]))

# Declare and fit the count vectorizer, transform the dataset
vectorizer = CountVectorizer(stop_words='english')
foo = vectorizer.fit_transform(data['STEMMED'])


