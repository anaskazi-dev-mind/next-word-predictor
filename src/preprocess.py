"""
preprocess.py

Purpose:
    Handles all text preprocessing required before training the LSTM and
    GRU next-word prediction models. Loads the raw corpus, cleans it,
    builds a tokenizer, generates n-gram training sequences, pads them,
    and saves the tokenizer for reuse during prediction.

Usage (run from project root):
    python -m src.preprocess
"""

import os
import pickle
from typing import List, Tuple

import numpy as np

# NOTE: Tokenizer is intentionally imported from tensorflow.keras, not from
# the standalone 'keras' package. Keras 3 completely removed the
# 'keras.preprocessing.text' module (including Tokenizer) -- it is only
# kept alive through TensorFlow's own tf.keras compatibility layer. The
# official replacement is 'keras.layers.TextVectorization', but that is a
# different tensor-based API (different methods, different workflow) and
# would require rewriting this file's logic, which is out of scope here.
from tensorflow.keras.preprocessing.text import Tokenizer

# pad_sequences uses the official, fully-migrated Keras 3 import path.
from keras.utils import pad_sequences

# ---------- Constants ----------
CORPUS_PATH = "data/corpus.txt"
TOKENIZER_SAVE_PATH = "models/tokenizer.pkl"


def load_corpus(file_path: str) -> List[str]:
    """
    Read the raw corpus file and return a list of non-empty sentences.

    Args:
        file_path: Path to the corpus text file.

    Returns:
        A list of raw sentence strings (one sentence per line).

    Raises:
        FileNotFoundError: If the corpus file does not exist.
        ValueError: If the corpus file is empty.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Corpus file not found at '{file_path}'. "
            "Make sure 'data/corpus.txt' exists and you are running "
            "this script from the project root."
        )

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    sentences = [line.strip() for line in lines if line.strip()]

    if not sentences:
        raise ValueError(f"Corpus file '{file_path}' is empty.")

    return sentences


def clean_sentence(sentence: str) -> str:
    """
    Normalize a sentence by lowercasing it.

    Punctuation is handled automatically by the Keras Tokenizer's default
    filters, so we intentionally keep this function minimal.

    Args:
        sentence: A single raw sentence.

    Returns:
        The lowercased, stripped sentence.
    """
    return sentence.lower().strip()


def build_tokenizer(sentences: List[str]) -> Tokenizer:
    """
    Fit a Keras Tokenizer on the given sentences.

    Args:
        sentences: List of cleaned sentences to fit the tokenizer on.

    Returns:
        A fitted Keras Tokenizer instance mapping words to integer indices.
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    return tokenizer


def generate_input_sequences(
    sentences: List[str], tokenizer: Tokenizer
) -> List[List[int]]:
    """
    Convert each sentence into progressively growing n-gram integer sequences.

    Example:
        Sentence tokens: [4, 5, 6, 7]
        Generated: [4, 5], [4, 5, 6], [4, 5, 6, 7]

    Args:
        sentences: List of cleaned sentences.
        tokenizer: A tokenizer already fitted on the corpus.

    Returns:
        A list of integer sequences of varying length.
    """
    input_sequences: List[List[int]] = []

    for sentence in sentences:
        token_list = tokenizer.texts_to_sequences([sentence])[0]

        # Skip sentences too short to form a valid (input, target) pair
        if len(token_list) < 2:
            continue

        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[: i + 1]
            input_sequences.append(n_gram_sequence)

    return input_sequences


def prepare_training_data(
    input_sequences: List[List[int]],
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Pad all sequences to the same length and split them into input (X)
    and target (y) arrays.

    Padding is applied at the start ('pre') so the target word always
    stays at the end, with the most recent context closest to it.

    Args:
        input_sequences: List of integer sequences of varying length.

    Returns:
        X: Padded input sequences (all columns except the last).
        y: Target word per sequence, as integer labels (not one-hot),
           to keep memory usage low. Models will consume these with
           'sparse_categorical_crossentropy'.
        max_sequence_length: Length used for padding; needed later during
           prediction to keep input shape consistent with training.

    Raises:
        ValueError: If no input sequences are provided.
    """
    if not input_sequences:
        raise ValueError("No input sequences were generated from the corpus.")

    max_sequence_length = max(len(seq) for seq in input_sequences)

    padded_sequences = pad_sequences(
        input_sequences, maxlen=max_sequence_length, padding="pre"
    )

    X = padded_sequences[:, :-1]
    y = padded_sequences[:, -1]

    return X, y, max_sequence_length


def save_tokenizer(tokenizer: Tokenizer, save_path: str) -> None:
    """
    Save the fitted tokenizer to disk using pickle, so the exact same
    word-to-index mapping can be reused during prediction without
    refitting on the corpus.

    Args:
        tokenizer: The fitted tokenizer to save.
        save_path: File path where the tokenizer will be saved.
    """
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with open(save_path, "wb") as file:
        pickle.dump(tokenizer, file)


if __name__ == "__main__":
    # Standalone test run to verify the preprocessing pipeline works
    # correctly before it is used by the training scripts.
    raw_sentences = load_corpus(CORPUS_PATH)
    cleaned_sentences = [clean_sentence(s) for s in raw_sentences]

    fitted_tokenizer = build_tokenizer(cleaned_sentences)
    vocab_size = len(fitted_tokenizer.word_index) + 1  # +1 for reserved index 0

    sequences = generate_input_sequences(cleaned_sentences, fitted_tokenizer)
    X, y, max_len = prepare_training_data(sequences)

    save_tokenizer(fitted_tokenizer, TOKENIZER_SAVE_PATH)

    print(f"Total sentences loaded: {len(cleaned_sentences)}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Total training sequences generated: {len(sequences)}")
    print(f"Max sequence length (for padding): {max_len}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Tokenizer saved to: {TOKENIZER_SAVE_PATH}")
