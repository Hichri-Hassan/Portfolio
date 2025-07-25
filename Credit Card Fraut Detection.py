import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import accuracy_score
  
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline # Using Pipeline from imblearn for convenience
card_data = pd.read_csv("C:/Users/hichr/OneDrive/Bureau/Kaagle/Titanic/creditcard.csv")
# Create target object and call it y
X = card_data.drop('Class', axis=1) # All columns except 'Class' 
y = card_data.Class
# 1. Split before SMOTE to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)

# 2. Apply scaling to training and test sets
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Apply SMOTE only on training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
card_model = RandomForestClassifier(
    n_estimators=1000,       # more trees = better performance (to a point)
    max_depth=30,            # limits tree depth to reduce overfittin
    random_state=1          # for reproducibility
)

card_model.fit(X_train_smote, y_train_smote)
y_preds = card_model.predict(X_test_scaled)
overall_accuracy = accuracy_score(y_test, y_preds)
print(f"Accuracy: {overall_accuracy:.4f}")
# 8. Save submission file
