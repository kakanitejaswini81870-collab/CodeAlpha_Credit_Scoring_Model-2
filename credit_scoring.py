import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')
print("="*70)
print(" " * 15 + "CREDIT SCORING MODEL - CODEALPHA TASK-2")
print("="*70)
# Step 1: Load and Explore Dataset
print("\n[STEP-1] Loading Dataset...")
df = pd.read_csv('dataset.csv')
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Missing Values:\n{df.isnull().sum()}")
# Step 2: Data Cleaning
print("\n[STEP-2] Cleaning Data...")
df = df.replace('NA', np.nan)
df['Saving accounts'].fillna('unknown', inplace=True)
df['Checking account'].fillna('unknown', inplace=True)
df['Credit amount'].fillna(df['Credit amount'].median(), inplace=True)
# Step 3: Feature Engineering
print("\n[STEP-3] Encoding Categorical Features...")
categorical_cols = ['Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(f"Total Features After Encoding: {df_encoded.shape[1] - 1}")
# Step 4: Define Features and Target
X = df_encoded.drop('Risk_good', axis=1)
y = df_encoded['Risk_good']
# Step 5: Split Dataset
print("\n[STEP-4] Splitting Train-Test Data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"Training Samples: {X_train.shape[0]} | Testing Samples: {X_test.shape[0]}")
# Step 6: Feature Scaling
print("\n[STEP-5] Scaling Features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Step 7: Train Logistic Regression
print("\n[STEP-6] Training Logistic Regression Model...")
lr_model = LogisticRegression(max_iter=2000, C=0.5, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_pred) * 100
print(f"Logistic Regression Accuracy: {lr_acc:.2f}%")
# Step 8: Train Random Forest
print("\n[STEP-7] Training Random Forest Model...")
rf_model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred) * 100
print(f"Random Forest Accuracy: {rf_acc:.2f}%")
# Step 9: Compare and Select Best Model
print("\n" + "="*70)
print(" " * 20 + "FINAL RESULTS")
print("="*70)
best_acc = max(lr_acc, rf_acc)
best_model_name = "Logistic Regression" if lr_acc > rf_acc else "Random Forest"
print(f"Best Performing Model: {best_model_name}")
print(f"Best Accuracy Score: {best_acc:.2f}%")
# Step 10: Detailed Evaluation
best_pred = lr_pred if lr_acc > rf_acc else rf_pred
print(f"\nDetailed Classification Report:\n{classification_report(y_test, best_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, best_pred)}")
# Step 11: Cross Validation
cv_score = cross_val_score(rf_model, X, y, cv=5).mean() * 100
print(f"\n5-Fold Cross Validation Score: {cv_score:.2f}%")
print("\n" + "="*70)
print(" " * 18 + "PROJECT COMPLETED SUCCESSFULLY")
print("="*70)
