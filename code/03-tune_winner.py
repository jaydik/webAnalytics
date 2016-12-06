from sklearn.linear_model import SGDClassifier
from time import time
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV

# Import data
X_train = joblib.load('../data/processed/X_train.pkl')
y_train = joblib.load('../data/processed/y_train.pkl')

# Build Model
svc_clf = SGDClassifier()

# Use gridsearchCV to find the best parameters
parameters = {
    'loss': ('hinge', 'log', 'modified_huber', 'squared_hinge'),
    'penalty': ('l1', 'l2', 'elasticnet'),
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
}

t0 = time()
tuned_clf = GridSearchCV(svc_clf, param_grid=parameters)
tuned_clf.fit(X_train, y_train)
print('Tuned and fit model in {:0.3f}'.format(time() - t0))

# Examine results
print(tuned_clf.grid_scores_)

# Dump model to disk
joblib.dump(tuned_clf, '../models/svc/tuned/svc_model.pkl')
