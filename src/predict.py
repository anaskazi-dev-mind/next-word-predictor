"""
predict.py

Purpose:
    Provides a shared prediction pipeline used by both the LSTM and GRU
    models. Loads a trained model and the saved tokenizer, preprocesses
    user input the same way the training corpus was processed, and
    returns the predicted next word (or a short generated continuation).

Usage (run from project root, for a standalone test):
    python -m src.predict
"""

import pickle
from typing import Optional

import numpy as np

# NOTE: Model (not Sequential) is used for type hints on loaded models.
# keras.models.load_model() can return a Sequential model, a Functional
# model, or a Model subclass -- its documented contract is the general
# 'Model' base class, not the more specific 'Sequential'. Since our
# models are always loaded (never constructed here), 'Model' is the
# technically accurate type. build_lstm_model()/build_gru_model() in the
# training scripts correctly keep 'Sequential' as their return type,
# since Sequential(...) genuinely returns a Sequential instance.
from keras.models import Model, load_model
from keras.utils import pad_sequences

# NOTE: Tokenizer type still imported from tensorflow.keras, for the same
# reason as in preprocess.py -- 'keras.preprocessing.text' (including
# Tokenizer) was removed entirely from the standalone Keras 3 package.
# This is only used here as a type hint; the actual Tokenizer object is
# created and saved by preprocess.py using this same class, so importing
# it from the same place keeps the type annotation accurate.
from tensorflow.keras.preprocessing.text import Tokenizer

# ---------- Constants ----------
LSTM_MODEL_PATH = "models/lstm_model.keras"
GRU_MODEL_PATH = "models/gru_model.keras"
TOKENIZER_PATH = "models/tokenizer.pkl"


def load_trained_model(model_path: str) -> Model:
    """
    Load a trained Keras model from disk.

    Args:
        model_path: Path to the saved '.keras' model file.

    Returns:
        The loaded Keras model, ready for inference.

    Raises:
        FileNotFoundError: If no model file exists at the given path.
    """
    try:
        return load_model(model_path)
    except (IOError, OSError) as error:
        raise FileNotFoundError(
            f"Could not load model from '{model_path}'. "
            "Make sure the model has been trained first "
            "(run 'python -m src.model_lstm' or 'python -m src.model_gru')."
        ) from error


def load_tokenizer(tokenizer_path: str) -> Tokenizer:
    """
    Load the saved tokenizer from disk.

    Args:
        tokenizer_path: Path to the saved '.pkl' tokenizer file.

    Returns:
        The fitted Tokenizer used during training.

    Raises:
        FileNotFoundError: If no tokenizer file exists at the given path.
    """
    try:
        with open(tokenizer_path, "rb") as file:
            return pickle.load(file)
    except (IOError, OSError) as error:
        raise FileNotFoundError(
            f"Could not load tokenizer from '{tokenizer_path}'. "
            "Make sure preprocessing/training has been run at least once."
        ) from error


def get_input_length(model: Model) -> int:
    """
    Determine the input sequence length a trained model expects, by
    reading it directly from the model's own input shape.

    This is deliberately used instead of recomputing the value from the
    training corpus. Recomputing from the corpus would silently produce
    a stale/incorrect length if 'data/corpus.txt' is ever edited without
    retraining the models, and would require predict.py to depend on the
    raw corpus and preprocessing pipeline for no real reason.

    Args:
        model: A trained LSTM or GRU model, loaded via load_trained_model().

    Returns:
        The expected input sequence length (number of timesteps).

    Raises:
        ValueError: If the model's input shape cannot be determined.
    """
    input_shape = model.input_shape  # e.g. (None, input_length)

    if input_shape is None or len(input_shape) < 2 or input_shape[1] is None:
        raise ValueError(
            f"Could not determine input length from the model's input "
            f"shape: {input_shape}. Make sure the model was built with a "
            "fixed input length (see build_lstm_model()/build_gru_model())."
        )

    return input_shape[1]


def predict_next_word(
    model: Model,
    tokenizer: Tokenizer,
    seed_text: str,
    input_length: int,
) -> Optional[str]:
    """
    Predict the single most likely next word given a seed text.

    Args:
        model: A trained LSTM or GRU model.
        tokenizer: The tokenizer used during training (for consistent
            word-to-index mapping).
        seed_text: The input text typed by the user.
        input_length: The input length the model was trained on
            (see get_input_length()).

    Returns:
        The predicted next word as a string, or None if the seed text
        contains no recognizable vocabulary words.
    """
    cleaned_text = seed_text.lower().strip()

    token_sequence = tokenizer.texts_to_sequences([cleaned_text])[0]

    # If none of the words in the seed text exist in the training
    # vocabulary, there is nothing meaningful to predict from.
    if not token_sequence:
        return None

    padded_sequence = pad_sequences(
        [token_sequence], maxlen=input_length, padding="pre"
    )

    predicted_probabilities = model.predict(padded_sequence, verbose=0)[0]
    predicted_index = int(np.argmax(predicted_probabilities))

    predicted_word = tokenizer.index_word.get(predicted_index)
    return predicted_word


def generate_text(
    model: Model,
    tokenizer: Tokenizer,
    seed_text: str,
    input_length: int,
    num_words: int = 5,
) -> str:
    """
    Generate a short continuation of the seed text by repeatedly
    predicting and appending the next word.

    Args:
        model: A trained LSTM or GRU model.
        tokenizer: The tokenizer used during training.
        seed_text: The starting text typed by the user.
        input_length: The input length the model was trained on.
        num_words: How many additional words to generate.

    Returns:
        The seed text extended with up to 'num_words' predicted words.
        Generation stops early if a word cannot be predicted.
    """
    current_text = seed_text

    for _ in range(num_words):
        next_word = predict_next_word(model, tokenizer, current_text, input_length)

        if next_word is None:
            break

        current_text += " " + next_word

    return current_text


if __name__ == "__main__":
    # Standalone test run to verify the prediction pipeline works
    # end-to-end using the already-trained LSTM model. Note that this no
    # longer touches the corpus or preprocess.py at all -- input_length
    # comes directly from the loaded model.
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    lstm_model = load_trained_model(LSTM_MODEL_PATH)
    input_length = get_input_length(lstm_model)

    test_seed = "deep learning is a"
    next_word = predict_next_word(lstm_model, tokenizer, test_seed, input_length)
    generated = generate_text(
        lstm_model, tokenizer, test_seed, input_length, num_words=5
    )

    print(f"Seed text: '{test_seed}'")
    print(f"Predicted next word: {next_word}")
    print(f"Generated continuation: {generated}")
