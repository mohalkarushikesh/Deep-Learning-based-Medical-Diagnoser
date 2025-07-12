# 🧠 Deep Learning-Based Medical Diagnoser using LSTM

## Overview

This project presents a deep learning model built using LSTM (Long Short-Term Memory) networks to diagnose diseases and recommend appropriate medications based on patient-reported symptoms. By processing sequential textual data, the model learns to identify patterns and relationships between symptoms, diagnoses, and treatments, revolutionizing healthcare with AI-powered decision support.

## ✨ Key Features

- 🩺 **Symptom Analysis**: Accepts natural language descriptions of patient issues.
- 📊 **LSTM-Based Classification**: Uses an RNN architecture to predict disease and medication.
- 💊 **Dual-Output Network**: Simultaneously generates diagnosis and prescription outputs.
- 📚 **End-to-End Pipeline**: Covers data preprocessing, model training, and prediction workflows.

## 🧰 Technologies Used

- **TensorFlow & Keras** – Deep learning framework.
- **Scikit-learn** – Label encoding utilities.
- **Pandas & NumPy** – Data manipulation and numerical operations.
- **LSTM Networks** – Designed for sequence data modeling.
- **Tokenizer & Embedding** – For transforming textual input into machine-readable form.

## 📦 Dataset Description

The dataset contains:

- **Patient_Problem**: Textual symptom descriptions
- **Disease**: Confirmed diagnosis
- **Prescription**: Recommended medications

_Source_: [Medical Dataset](https://raw.githubusercontent.com/adil200/Medical-Diagnoser/main/medical_data.csv)

## 🛠 Model Architecture

```text
Input Layer → Embedding Layer → LSTM Layer
          ↘                         ↙
   Dense (Disease Output)   Dense (Prescription Output)
```

- Embedding size: 64
- LSTM units: 64
- Output layers use Softmax for multi-class classification

## 🧪 Training Details

- Loss: Categorical crossentropy
- Optimizer: Adam
- Epochs: 100
- Batch Size: 32
- Metrics: Accuracy for both outputs

## 🚀 How to Run

```bash
# Clone the repo and install dependencies
pip install tensorflow pandas numpy scikit-learn

# Load and preprocess data
python train_model.py

# Make predictions
python predict.py
```

## 📈 Example Prediction

Input:
```text
"I've experienced a loss of appetite and don't enjoy food anymore."
```

Output:
```
Predicted Disease: Depression
Suggested Prescription: Fluoxetine
```

## 🌱 Future Enhancements

- 🧾 **Integration with Electronic Health Records (EHRs)** for richer contextual data.
- 🗣️ **Voice-to-Text Input** for hands-free patient interaction.
- 📱 **Mobile App Interface** for real-time diagnosis and medication suggestions.
- 🌐 **Multilingual Support** to handle symptom descriptions in regional languages.
- 🔍 **Explainability Layer** to show symptom-disease correlations for transparency.
- 🤝 **Clinician Feedback Loop** to fine-tune predictions with expert corrections.
- 🧬 **Genomic & Lab Data Integration** for precision diagnostics.

## ⚠️ Disclaimer

This tool is designed for educational and research purposes only. It is not intended for clinical use or to replace medical professionals.
