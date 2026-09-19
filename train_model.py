import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("student_performance.csv")

# Basic cleaning
numeric_columns = [
    "Age", "Attendance", "Study_Hours", "Previous_Marks",
    "Assignment_Score", "Participation", "Previous_Exam_Score",
    "Sleep_Hours", "Final_Performance"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

categorical_columns = [
    "Gender", "Department", "Internet_Access",
    "Extracurricular_Activity", "Parental_Support"
]

for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Features used by the prediction system
feature_columns = [
    "Age", "Gender", "Department", "Attendance", "Study_Hours",
    "Previous_Marks", "Assignment_Score", "Participation",
    "Previous_Exam_Score", "Sleep_Hours", "Internet_Access",
    "Extracurricular_Activity", "Parental_Support"
]

X = df[feature_columns]
y = df["Final_Performance"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

numeric_features = [
    "Age", "Attendance", "Study_Hours", "Previous_Marks",
    "Assignment_Score", "Participation", "Previous_Exam_Score",
    "Sleep_Hours"
]

categorical_features = [
    "Gender", "Department", "Internet_Access",
    "Extracurricular_Activity", "Parental_Support"
]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

model = RandomForestRegressor(
    n_estimators=250,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("==========================================")
print(" STUDENT PERFORMANCE PREDICTION - MODEL")
print("==========================================")
print(f"Dataset shape : {df.shape}")
print(f"Training rows : {len(X_train)}")
print(f"Testing rows  : {len(X_test)}")
print(f"MAE           : {mae:.2f}")
print(f"RMSE          : {rmse:.2f}")
print(f"R2 Score      : {r2:.2f}")

model_data = {
    "pipeline": pipeline,
    "features": feature_columns,
    "metrics": {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 2)
    }
}

with open("student_performance_model.pkl", "wb") as file:
    pickle.dump(model_data, file)

print("\nModel saved successfully!")
