# Pokemon Type AI

A deep learning project that predicts a Pokémon's primary type using its battle stats and physical attributes.

## Features

- Neural Network built with TensorFlow/Keras
- Data preprocessing using scikit-learn
- Label encoding and feature scaling
- Training visualization with Matplotlib
- Live prediction system using custom inputs
- Model saving for future use

---

## Dataset

This project uses a Pokémon dataset collected from Kaggle.

Dataset includes:
- HP
- Attack
- Defense
- Special Attack
- Special Defense
- Speed
- Height
- Weight
- Primary Pokémon Type

---

## Technologies Used

- Python
- TensorFlow
- Pandas
- NumPy
- Matplotlib
- scikit-learn

---

## Model Architecture

Input Layer → Dense(128) → Dropout → Dense(64) → Dense(32) → Softmax Output

---

## Training Results

The model was trained for 50 epochs using the Adam optimizer and sparse categorical crossentropy loss.

Example metrics:
- Validation Accuracy: XX%

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

---


## Author

Mohamed Amdjed Dariadi