import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn import metrics
from imblearn.combine import SMOTEENN
from collections import Counter
import joblib

# 1. Load dataset
df = pd.read_csv("pima.csv")

# 2. Replace 0s with NaN for specific columns
zero_columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_columns] = df[zero_columns].replace(0, np.nan)

# 3. Impute missing values with median
imputer = SimpleImputer(strategy='median')
df[zero_columns] = imputer.fit_transform(df[zero_columns])

# 4. Features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
feature_names = X.columns.tolist()

# 5. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Handle class imbalance
smote_enn = SMOTEENN(random_state=1)
X_resampled, y_resampled = smote_enn.fit_resample(X_scaled, y)
print("Before augmentation:", Counter(y))
print("After augmentation:", Counter(y_resampled))

# 7. Split into train/test sets
x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=1)

# 8. Hyperparameter tuning with GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear']
}

grid = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid.fit(x_train, y_train)

# 9. Best model from grid search
reg = grid.best_estimator_
print("Best Parameters:", grid.best_params_)

# 10. Evaluation
y_pred = reg.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy on test set:", accuracy)

# 11. Save model, scaler, and feature names
joblib.dump(reg, "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_names, "feature_names.pkl")
print(" Model and artifacts saved successfully.")
