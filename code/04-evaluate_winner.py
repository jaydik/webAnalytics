from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Import models
clf = joblib.load('../models/svc/tuned/svc_model.pkl')

# Import Data
X_test = joblib.load('../data/processed/X_test.pkl')
y_test = joblib.load('../data/processed/y_test.pkl')

# Evaluate RF
pred = clf.predict(X_test)

print('Tuned Model Scores\n\n')
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))
print(accuracy_score(y_test, pred))
print('-' * 80)