# Data-science-project
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('patient_data.csv')

# Encode categorical features
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Separate features and target
X = df.drop('disease', axis=1)
y = df['disease']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Predict a new patient (example)
new_patient = pd.DataFrame([{
    'age': 45,
    'gender': label_encoders['gender'].transform(['Male'])[0],
    'fever': 1,
    'cough': 1,
    'headache': 0,
    'fatigue': 1
}])

prediction = model.predict(new_patient)
predicted_disease = label_encoders['disease'].inverse_transform(prediction)[0]
print(f"Predicted Disease: {predicted_disease}")
