# train_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# === Load Data ===
df = pd.read_csv("emotions.csv")
X = df.drop(columns=["label"])
y = df["label"]

# === Encode Labels ===
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# === Normalize EEG Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# === Train Neural Net Classifier ===
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=300,
    random_state=42
)
mlp.fit(X_train, y_train)

# === Evaluate Model ===
y_pred = mlp.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("✅ Accuracy:", acc)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# === Save Model and Tools ===
joblib.dump(mlp, "emotion_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("💾 Model, Scaler, and LabelEncoder saved!")
