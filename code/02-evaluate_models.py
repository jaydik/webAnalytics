from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Import models
rf_clf = joblib.load('../models/rf/rf_model.pkl')
nb_clf = joblib.load('../models/nb/nb_model.pkl')
svc_clf = joblib.load('../models/svc/svc_model.pkl')

# Import Data
X_test = joblib.load('../data/processed/X_test.pkl')
y_test = joblib.load('../data/processed/y_test.pkl')

# Evaluate RF
rf_pred = rf_clf.predict(X_test)

print('Random Forest\n\n')
print(classification_report(y_test, rf_pred))
print(confusion_matrix(y_test, rf_pred))
print(accuracy_score(y_test, rf_pred))
print('-' * 80)

# Evaluate NB
nb_pred = nb_clf.predict(X_test)

print('Naive Bayes\n\n')
print(classification_report(y_test, nb_pred))
print(confusion_matrix(y_test, nb_pred))
print(accuracy_score(y_test, nb_pred))
print('-' * 80)

# Evaluate NB
svc_pred = svc_clf.predict(X_test)

print('SVM\n\n')
print(classification_report(y_test, svc_pred))
print(confusion_matrix(y_test, svc_pred))
print(accuracy_score(y_test, svc_pred))
print('-' * 80)
