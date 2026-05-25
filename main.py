import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ==============================
# Load Dataset
# ==============================

DATASET_PATH = "pokemon_complete_2025.csv"

try:
    df = pd.read_csv(DATASET_PATH)
    print("Dataset loaded successfully.\n")

except Exception as error:
    print(f"Error loading dataset: {error}")
    exit()


# ==============================
# Feature Selection
# ==============================

features = [
    "hp",
    "attack",
    "defense",
    "sp_attack",
    "sp_defense",
    "speed",
    "height_m",
    "weight_kg"
]

X = df[features].fillna(0)
y = df["type_1"]


# ==============================
# Encode Labels
# ==============================

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)


# ==============================
# Train / Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)


# ==============================
# Data Scaling
# ==============================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==============================
# Build Neural Network
# ==============================

model = Sequential([
    Dense(128, activation="relu", input_shape=(len(features),)),
    Dropout(0.2),

    Dense(64, activation="relu"),
    Dense(32, activation="relu"),

    Dense(len(encoder.classes_), activation="softmax")
])


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==============================
# Training
# ==============================

print("Training model...\n")

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)


# ==============================
# Model Evaluation
# ==============================

loss, accuracy = model.evaluate(X_test, y_test)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")


# ==============================
# Save Model
# ==============================

model.save("pokemon_type_model.h5")

print("Model saved successfully.\n")


# ==============================
# Training Visualization
# ==============================

plt.figure(figsize=(10, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Training Progress")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)

plt.show()


# ==============================
# Live Prediction
# ==============================

print("\n" + "=" * 40)
print("Pokemon Type Prediction")
print("=" * 40)

try:
    hp = float(input("HP: "))
    attack = float(input("Attack: "))
    defense = float(input("Defense: "))
    sp_attack = float(input("Special Attack: "))
    sp_defense = float(input("Special Defense: "))
    speed = float(input("Speed: "))
    height = float(input("Height (m): "))
    weight = float(input("Weight (kg): "))

    sample = scaler.transform([[
        hp,
        attack,
        defense,
        sp_attack,
        sp_defense,
        speed,
        height,
        weight
    ]])

    prediction = model.predict(sample)

    predicted_index = np.argmax(prediction)
    predicted_type = encoder.inverse_transform([predicted_index])[0]

    confidence = np.max(prediction) * 100

    print(f"\nPredicted Type: {predicted_type}")
    print(f"Confidence: {confidence:.2f}%")

except ValueError:
    print("Please enter valid numeric values.")