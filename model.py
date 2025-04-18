import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Final_Modified_Project.csv")

# Handling missing values
df.dropna(inplace=True)

# Encode categorical features
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separate features and target variable
X = df.drop(columns=['Career'])  # Adjust column name if needed
y = df['Career']

# Feature Selection
vt = VarianceThreshold(threshold=0.01)
X_vt = vt.fit_transform(X)

skb = SelectKBest(score_func=f_classif, k=10)
X_selected = skb.fit_transform(X_vt, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train XGBoost model
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model & preprocessing objects
joblib.dump(model, "career_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(vt, "vt.pkl")
joblib.dump(skb, "skb.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("Model training complete and saved successfully.")
