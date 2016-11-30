import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

data = pd.read_csv('../data/raw/uci-news-aggregator.csv')
category = data['CATEGORY']
data = data.drop('CATEGORY', axis=1)

X_train, X_test, y_train, y_test = train_test_split(data, category, test_size=0.25, random_state=8675309)

X_train.to_csv('../data/processed/X_train.csv', index=False)
X_test.to_csv('../data/processed/X_test.csv', index=False)
y_train.to_csv('../data/processed/y_train.csv', index=False)
y_test.to_csv('../data/processed/y_test.csv', index=False)
