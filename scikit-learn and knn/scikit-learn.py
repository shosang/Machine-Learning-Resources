# -*- coding: utf-8 -*-
"""
scikit-learn: k nearest neighbours - classify unknown data point based on its distance from known points
Data Source: UCI Machine Learning Repository - Congressional Voting Records
    https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records
"""

import pandas as pd
import yaml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

with open("scikit-learn.yaml", 'r') as stream:
    try:
        yml = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)


df = pd.read_csv(yml["data"], header=None, names=yml["col_names"])
df.replace({'?':0, 'n':0, 'y':1}, inplace=True)

## Explore dataframe
#df.head(20)
#df.info()
#df.describe()

# Create NumPy arrays for the features (X) and the target variable (y)
y = df['party'].values
X = df.drop('party', axis=1).values

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.25, 
                                                    random_state=100, 
                                                    stratify=y)

# ____________________________________________________________________________________________________

# Elbow plot
acc_val = []
for k in range(1,11,2):
    k = k+1
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)  
    acc_val.append(knn.score(X_test, y_test))
    
plt.title('k-NN: Elbow Plot')
plt.plot(range(1,11,2), acc_val, label = 'Test Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

# Fit k-Neighest Neighbours manually
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)

# Predict the labels for the training data X: y_pred
y_pred = knn.predict(X_test)

# Print the accuracy
print("Accuracy: {}".format(knn.score(X_test, y_test)))

# Confusion Matrix
print("Confusion Matrix:\n {}".format(confusion_matrix(y_test, y_pred)))

# Classification Report: Precision, Recal, F1-Score, Support
print("Classification Report: \n {}".format(classification_report(y_test, y_pred)))

# ROC Curve
y_pred_prob = knn.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label=knn.classes_[1])
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# AUC Score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# ____________________________________________________________________________________________________

# Perform Grid Search
grid_params = {
        'n_neighbors': [3,5,7,9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
        }

gs = GridSearchCV(KNeighborsClassifier(),
                  grid_params,
                  verbose=1,
                  cv=5,
                  n_jobs=-1)

gs_results = gs.fit(X_train, y_train)
print("Grid Search Best Score: {}".format(gs_results.best_score_))
print("Grid Search Best Estimator: {}".format(gs_results.best_estimator_))
print("Grid Search Best Parameters: {}".format(gs_results.best_params_))

# Predict the labels for the training data X: y_pred
gs_y_pred = gs_results.predict(X_test)

# Print the accuracy
print("Grid Search Accuracy: {}".format(gs_results.score(X_test, y_test)))

# Confusion Matrix
print("Grid Search Confusion Matrix:\n {}".format(confusion_matrix(y_test, gs_y_pred)))

# Classification Report: Precision, Recal, F1-Score, Support
print("Grid Search Classification Report: \n {}".format(classification_report(y_test, gs_y_pred)))

# ROC Curve
gs_y_pred_prob = gs_results.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, gs_y_pred_prob, pos_label=gs_results.classes_[1])
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Grid Search')
plt.show()

# AUC Score - 0.0003 improvement in AUC
print("Grid Seach AUC: {}".format(roc_auc_score(y_test, gs_y_pred_prob)))
