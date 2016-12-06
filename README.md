# Web Analytics Project - Jon Dickerson
## Kaggle News Article Classification
### 2016 Dec 06

If you're running with a different version of scikit (namely 18) then the modules have changed a bit.
GridSearchCV (used in 03-tune_winner.py) won't work because they moved where that module is.

You can fix this by doing the following:
  - Changing "from sklearn.grid_search import GridSearchCV" to "from sklearn.model_selection import GridSearchCV"
  - Disabling the print statement "print(tuned_clf.grid_scores_)", as the attribute has changed names