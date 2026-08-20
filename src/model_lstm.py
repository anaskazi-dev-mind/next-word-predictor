"""
model_lstm.py

Purpose:
    Defines, trains, and saves the LSTM-based next-word prediction model.
    Reuses the preprocessing pipeline from preprocess.py so that both the
    LSTM and GRU models are trained on identical data and tokenization.

Usage (run from project root):
    python -m src.model_lstm
"""

import os
from typing import Tuple

# Fully migrated to Keras 3's standalone import paths.
from keras.models import Sequential
from keras.layers import Input, Embedding, LSTM, Dense
from keras.callbacks import EarlyStopping

from src.preprocess import (
    CORPUS_PATH,
    TOKENIZER_SAVE_PATH,
    load_corpus,
    clean_sentence,
    build_tokenizer,
    generate_input_sequences,
    prepare_training_data,
    save_tokenizer,
)

# ---------- Constants ----------
MODEL_SAVE_PATH = "models/lstm_model.keras"
EMBEDDING_DIM = 100
LSTM_UNITS = 150
EPOCHS = 100
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 5


def build_lstm_model(vocab_size: int, input_length: int) -> Sequential:
    """
    Build and compile the LSTM next-word prediction model.

    Architecture:
        Input -> Embedding -> LSTM -> Dense (softmax over vocabulary)

    Args:
        vocab_size: Total number of unique words in the vocabulary
            (including the reserved 0 index).
        input_length: Length of each padded input sequence.

    Returns:
        A compiled Keras Sequential model ready for training.
    """
    model = Sequential(
        [
            Input(shape=(input_length,)),
            Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM),
            LSTM(LSTM_UNITS),
            Dense(vocab_size, activation="softmax"),
        ]
    )

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    return model


def train_lstm_model() -> Tuple[Sequential, int]:
    """
    Run the full pipeline: load data, preprocess it, build the LSTM model,
    train it, and save both the model and tokenizer to disk.

    Returns:
        A tuple of (trained model, vocabulary size) for reference/logging.
    """
    # Step 1: Load and clean the corpus
    raw_sentences = load_corpus(CORPUS_PATH)
    cleaned_sentences = [clean_sentence(sentence) for sentence in raw_sentences]

    # Step 2: Build tokenizer and generate training sequences
    tokenizer = build_tokenizer(cleaned_sentences)
    vocab_size = len(tokenizer.word_index) + 1  # +1 for reserved index 0

    sequences = generate_input_sequences(cleaned_sentences, tokenizer)
    X, y, max_sequence_length = prepare_training_data(sequences)
    input_length = X.shape[1]

    # Step 3: Build the model
    model = build_lstm_model(vocab_size=vocab_size, input_length=input_length)
    model.summary()

    # Step 4: Train with early stopping to avoid overfitting/wasted compute
    early_stopping = EarlyStopping(
        monitor="loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )

    model.fit(
        X,
        y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1,
    )

    # Step 5: Save the trained model
    save_dir = os.path.dirname(MODEL_SAVE_PATH)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    model.save(MODEL_SAVE_PATH)

    # Step 6: Save the tokenizer as well (safe to overwrite -- identical
    # tokenizer will be produced by GRU training too, since both use the
    # same corpus and preprocessing logic)
    save_tokenizer(tokenizer, TOKENIZER_SAVE_PATH)

    print(f"\nLSTM model saved to: {MODEL_SAVE_PATH}")
    print(f"Tokenizer saved to: {TOKENIZER_SAVE_PATH}")

    return model, vocab_size


if __name__ == "__main__":
    train_lstm_model()
