# evaluate_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

# === Load data and model ===
df = pd.read_csv("emotions.csv")
X = df.drop(columns=["label"])
y = df["label"]

label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")
model = joblib.load("emotion_model.pkl")

# === Encode and normalize ===
y_encoded = label_encoder.transform(y)
X_scaled = scaler.transform(X)

# === Split same as before ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# === Predict and evaluate ===
y_pred = model.predict(X_test)

print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap="Blues")
plt.title("Emotion Prediction - Confusion Matrix")
plt.show()
