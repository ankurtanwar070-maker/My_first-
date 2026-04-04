import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Ensure src directory exists
os.makedirs('src', exist_ok=True)

# Create sample dataset (similar to Bank Marketing dataset)
np.random.seed(42)
n_samples = 4000

data = {
    'age': np.random.randint(18, 95, n_samples),
    'job': np.random.choice(['admin.', 'technician', 'services', 'management', 'retired', 
                             'blue-collar', 'unemployed', 'entrepreneur', 'housemaid', 
                             'unknown', 'self-employed', 'student'], n_samples),
    'marital': np.random.choice(['married', 'single', 'divorced'], n_samples),
    'education': np.random.choice(['unknown', 'secondary', 'primary', 'tertiary'], n_samples),
    'default': np.random.choice(['no', 'yes'], n_samples, p=[0.8, 0.2]),
    'balance': np.random.randint(-1000, 5000, n_samples),
    'housing': np.random.choice(['no', 'yes'], n_samples, p=[0.4, 0.6]),
    'loan': np.random.choice(['no', 'yes'], n_samples, p=[0.7, 0.3]),
    'contact': np.random.choice(['unknown', 'telephone', 'cellular'], n_samples),
    'day': np.random.randint(1, 32, n_samples),
    'month': np.random.choice(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                               'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], n_samples),
    'duration': np.random.randint(0, 4000, n_samples),
    'campaign': np.random.randint(1, 10, n_samples),
    'pdays': np.random.randint(-1, 400, n_samples),
    'previous': np.random.randint(0, 10, n_samples),
    'poutcome': np.random.choice(['unknown', 'other', 'failure', 'success'], n_samples),
    'deposit': np.random.choice(['no', 'yes'], n_samples, p=[0.8, 0.2])
}

df = pd.DataFrame(data)

print("Dataset created successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Separate features and target
X = df.drop('deposit', axis=1)
y = df['deposit']

# Encode categorical variables
le_dict = {}
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

# Encode target
le_target = LabelEncoder()
y = le_target.fit_transform(y)

# Scale numerical features
scaler = StandardScaler()
numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Train Random Forest model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training Accuracy: {train_score:.4f}")
print(f"Testing Accuracy: {test_score:.4f}")

# Save model, scaler, and label encoder
joblib.dump(model, 'src/marketing_response_model.pkl')
joblib.dump(scaler, 'src/scaler.pkl')
joblib.dump(le_dict, 'src/label_encoder.pkl')

print("\nModel files saved successfully!")
print("✓ src/marketing_response_model.pkl")
print("✓ src/scaler.pkl")
print("✓ src/label_encoder.pkl")
