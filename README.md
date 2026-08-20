# 🔤 Next-Word Prediction using LSTM & GRU

A deep learning project that predicts the **next word** in a sentence using two Recurrent Neural Network architectures — **LSTM (Long Short-Term Memory)** and **GRU (Gated Recurrent Unit)**.

The project trains both models on the **same custom dataset** using the **same preprocessing pipeline** and compares their predictions through an interactive **Streamlit web application**.

---

## 🌐 Live Demo

**Try the app here:**  
https://next-word-predictor-01.streamlit.app/

---

## 📂 GitHub Repository

https://github.com/anaskazi-dev-mind/next-word-predictor

---

## ✨ Features

- Compare **LSTM** and **GRU** side by side
- Predict the next word from a user-provided sentence
- Generate a short continuation of the sentence
- Shared tokenizer for fair comparison
- Interactive Streamlit interface
- Cached model loading for faster predictions
- Clean and modular project structure

---

## 🧠 Models Used

### LSTM

Long Short-Term Memory networks are designed to learn long-range dependencies in sequential data and are widely used in Natural Language Processing.

### GRU

Gated Recurrent Units are a simplified version of LSTM with fewer parameters while maintaining competitive performance.

Both models were trained using identical preprocessing so that only the recurrent architecture changes.

---

## 📚 Dataset

The project uses a **custom corpus** created for educational purposes.

The dataset contains sentences from two categories:

- Artificial Intelligence & Machine Learning
- General English

This combination helps the models learn both technical vocabulary and everyday sentence patterns.

Example sentences:

```text
Deep learning is a subset of machine learning.
Natural language processing is a field of artificial intelligence.
The weather is pleasant today.
Reading before bed helps me relax.
Programming helps solve real world problems.
```

---

## ⚙️ Model Configuration

Both models use the same configuration.

| Parameter | Value |
|-----------|------:|
| Embedding Dimension | 100 |
| Hidden Units | 150 |
| Batch Size | 32 |
| Maximum Epochs | 100 |
| Early Stopping | Enabled |

Architecture:

```text
Input
   │
Embedding (100)
   │
LSTM / GRU (150 Units)
   │
Dense (Softmax)
   │
Predicted Next Word
```

---

## 🛠 Tech Stack

- Python
- TensorFlow
- Keras
- NumPy
- Streamlit

---

## 📁 Project Structure

```text
next-word-predictor/
│
├── app.py
├── requirements.txt
├── runtime.txt
│
├── data/
│   └── corpus.txt
│
├── models/
│   ├── lstm_model.keras
│   ├── gru_model.keras
│   └── tokenizer.pkl
│
└── src/
    ├── preprocess.py
    ├── model_lstm.py
    ├── model_gru.py
    ├── predict.py
    └── __init__.py
```

---

## 🚀 Installation

Clone the repository.

```bash
git clone https://github.com/anaskazi-dev-mind/next-word-predictor.git

cd next-word-predictor
```

Create a virtual environment.

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 💻 Example

Input

```text
deep learning is a
```

Output

```text
Predicted Next Word:
subset

Generated Text:
deep learning is a subset of machine learning models
```

---

## 📖 How It Works

1. Load the trained models and tokenizer.
2. Convert user input into token sequences.
3. Pad the sequence to the required input length.
4. Pass the sequence to both models.
5. Predict the most probable next word.
6. Repeat prediction to generate a short continuation.
7. Display both model outputs for comparison.

---

## 🎯 Learning Objectives

This project demonstrates:

- Text preprocessing
- Tokenization
- Sequence generation
- Padding
- Word embeddings
- Language modeling
- LSTM networks
- GRU networks
- Text generation
- Model comparison
- Streamlit deployment

---

## 📌 Future Improvements

- Larger training corpus
- Top-k predictions
- Prediction probabilities
- Transformer-based models
- Beam search text generation
- Model evaluation metrics
- Training visualizations
- User-uploaded datasets

---

## 👨‍💻 Author

**Anas Kazi**

GitHub: https://github.com/anaskazi-dev-mind

---

## 📄 License

This project is licensed under the **MIT License**.

---

## ⭐ If You Like This Project

If you found this project useful, consider giving it a ⭐ on GitHub.