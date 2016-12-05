from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from time import time

# Import data
X_train = joblib.load('../data/processed/X_train.pkl')
y_train = joblib.load('../data/processed/y_train.pkl')

# Build Model
print('Training Random Forest')
t0 = time()
rf_clf = RandomForestClassifier(random_state=8675309)
rf_clf.fit(X_train, y_train)
print('Random Forest Training Time {:0.3f}'.format(time() - t0))
print('-' * 80)
print('Training Naive Bayes')
t0 = time()
nb_clf = MultinomialNB()
nb_clf.fit(X_train, y_train)
print('Naive Bayes Training Time {:0.3f}'.format(time() - t0))
print('-' * 80)
print('Training SVM')
t0 = time()
svm_clf = SGDClassifier(random_state=8675309)
svm_clf.fit(X_train, y_train)
print('SVM Training Time {:0.3f}'.format(time() - t0))
print('-' * 80)

# Dump model to disk
joblib.dump(rf_clf, '../models/rf/rf_model.pkl')
joblib.dump(nb_clf, '../models/nb/nb_model.pkl')
joblib.dump(svm_clf, '../models/svc/svc_model.pkl')
