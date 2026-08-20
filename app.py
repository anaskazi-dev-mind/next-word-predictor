"""
app.py

Purpose:
    Streamlit web application for the Next-Word Prediction project.
    Lets the user type the start of a sentence and see the predicted
    next word -- and a short generated continuation -- from both the
    LSTM and GRU models, side by side.

Usage (run from project root):
    streamlit run app.py
"""

from typing import Optional

import streamlit as st

from src.predict import (
    LSTM_MODEL_PATH,
    GRU_MODEL_PATH,
    TOKENIZER_PATH,
    load_trained_model,
    load_tokenizer,
    get_input_length,
    predict_next_word,
    generate_text,
)

# Imported for display purposes only (sidebar + architecture summary), so
# these values can never drift out of sync with the actual training
# configuration. Importing a module only reads its top-level constants
# and function definitions -- it does not execute any training, since
# both files guard their training calls behind `if __name__ == "__main__":`.
from src.model_lstm import EMBEDDING_DIM, LSTM_UNITS, EPOCHS, BATCH_SIZE
from src.model_gru import GRU_UNITS

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="Next-Word Prediction: LSTM vs GRU",
    page_icon="🔤",
    layout="centered",
)


# ---------- Cached Resource Loading ----------
@st.cache_resource
def load_resources():
    """
    Load both trained models and the shared tokenizer once per app
    session, and cache them.

    Using st.cache_resource (rather than st.cache_data) is deliberate:
    it is the correct Streamlit primitive for objects like ML models
    that should be loaded once and reused across reruns, instead of
    being copied/serialized on every access.

    Returns:
        A tuple of (lstm_model, gru_model, tokenizer, input_length).

    Raises:
        FileNotFoundError: If the trained models/tokenizer are missing.
    """
    lstm_model = load_trained_model(LSTM_MODEL_PATH)
    gru_model = load_trained_model(GRU_MODEL_PATH)
    tokenizer = load_tokenizer(TOKENIZER_PATH)

    # Both models were trained on identical preprocessing, so they share
    # the same input length -- reading it from one is sufficient.
    input_length = get_input_length(lstm_model)

    return lstm_model, gru_model, tokenizer, input_length


# ---------- Sidebar ----------
def render_sidebar() -> None:
    """Render the project context sidebar (overview, stack, dataset, etc.)."""
    with st.sidebar:
        st.header("📘 About This Project")
        st.write(
            "A next-word prediction app that trains two recurrent neural "
            "networks — an **LSTM** and a **GRU** — on the same text "
            "corpus and configuration, so their predictions can be "
            "compared side by side."
        )

        st.divider()

        st.subheader("🛠️ Technologies Used")
        st.markdown("- TensorFlow\n- Keras\n- Streamlit\n- LSTM\n- GRU")

        st.divider()

        st.subheader("📚 Dataset")
        st.write("Custom corpus containing AI/ML and general English sentences.")

        st.divider()

        st.subheader("🧠 Model Architecture")
        st.markdown(
            "Both models share an identical pipeline for a fair "
            "comparison — only the recurrent layer differs:"
        )
        st.markdown(
            f"`Input → Embedding({EMBEDDING_DIM}) → "
            f"LSTM({LSTM_UNITS}) / GRU({GRU_UNITS}) → Dense(softmax)`"
        )

        st.divider()

        st.subheader("⚙️ Model Configuration")
        st.markdown(
            f"- **Embedding dimension:** {EMBEDDING_DIM}\n"
            f"- **LSTM hidden units:** {LSTM_UNITS}\n"
            f"- **GRU hidden units:** {GRU_UNITS}\n"
            f"- **Max epochs:** {EPOCHS} (with early stopping)\n"
            f"- **Batch size:** {BATCH_SIZE}"
        )

        st.divider()

        st.subheader("👤 Author")
        st.write("Anas Kazi")

        st.subheader("🔗 Source Code")
        st.markdown(
            "[View on GitHub](https://github.com/anaskazi-dev-mind/next-word-predictor)"
        )


# ---------- Header ----------
def render_header() -> None:
    """Render the app title, tagline, and a short explanatory info box."""
    st.title("🔤 Next-Word Prediction")
    st.caption("Compare how an LSTM and a GRU model complete the same sentence.")

    st.info(
        "This app trains two recurrent neural networks — an **LSTM** and "
        "a **GRU** — on an identical dataset and configuration. Type the "
        "start of a sentence below to see how each model predicts what "
        "comes next.",
        icon="ℹ️",
    )


# ---------- Input Form ----------
def render_input_form():
    """
    Render the prediction input form.

    Wrapping the inputs in st.form means the app only reruns when the
    user submits (click or Enter), not on every keystroke or slider
    drag -- this avoids unnecessary reruns while typing.

    Returns:
        A tuple of (seed_text, num_words, submitted).
    """
    st.subheader("✍️ Try It Yourself")

    with st.form("prediction_form", border=True):
        seed_text = st.text_input(
            "Start of your sentence",
            placeholder="e.g. deep learning is a",
            help="Type a few words in English. Both models will predict what comes next.",
        )
        num_words = st.slider(
            "Words to generate",
            min_value=1,
            max_value=10,
            value=5,
            help="How many additional words each model should generate after the first prediction.",
        )
        submitted = st.form_submit_button("Predict Next Word", use_container_width=True)

    return seed_text, num_words, submitted


# ---------- Result Card ----------
def render_model_card(
    column,
    label: str,
    icon: str,
    model,
    tokenizer,
    seed_text: str,
    input_length: int,
    num_words: int,
) -> Optional[str]:
    """
    Render a single model's prediction results inside a bordered card
    within the given Streamlit column.

    Args:
        column: The st.columns() slot to render into.
        label: Display name for the model ("LSTM" or "GRU").
        icon: A small emoji used to visually distinguish this card.
        model: A trained LSTM or GRU model.
        tokenizer: The shared tokenizer.
        seed_text: The user's input text.
        input_length: The input length the model expects.
        num_words: How many words to generate for the continuation.

    Returns:
        The predicted next word, or None if no prediction was possible.
    """
    with column:
        with st.container(border=True):
            st.subheader(f"{icon} {label}")

            next_word = predict_next_word(model, tokenizer, seed_text, input_length)

            if next_word is None:
                st.warning(
                    "None of these words were seen during training. Try different words."
                )
                return None

            st.metric("Predicted next word", next_word)

            generated = generate_text(
                model, tokenizer, seed_text, input_length, num_words
            )
            st.caption("Generated Text")
            st.text_area(
                "Generated text",
                value=generated,
                height=80,
                disabled=True,
                label_visibility="collapsed",
            )

            return next_word


# ---------- Comparison Summary ----------
def render_comparison(lstm_word: Optional[str], gru_word: Optional[str]) -> None:
    """
    Render a short summary highlighting whether the two models agreed
    on the predicted next word.

    Args:
        lstm_word: The word predicted by the LSTM model (or None).
        gru_word: The word predicted by the GRU model (or None).
    """
    st.subheader("Prediction Summary")

    if lstm_word is None or gru_word is None:
        st.caption(
            "Comparison unavailable — one or both models could not make a prediction."
        )
        return

    with st.container(border=True):
        if lstm_word == gru_word:
            st.success("Both models predicted the same next word.")
            st.metric("Predicted word", lstm_word)
        else:
            st.info("The models produced different predictions.")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("LSTM", lstm_word)
            with col_b:
                st.metric("GRU", gru_word)


# ---------- Main App ----------
def main() -> None:
    render_sidebar()
    render_header()

    try:
        lstm_model, gru_model, tokenizer, input_length = load_resources()
    except FileNotFoundError:
        st.error(
            "Trained models were not found. Please train them first by running, "
            "from the project root:\n\n"
            "```\npython -m src.model_lstm\npython -m src.model_gru\n```"
        )
        st.stop()

    st.divider()

    seed_text, num_words, submitted = render_input_form()

    if submitted:
        if not seed_text.strip():
            st.warning("Please type some text before predicting.")
        else:
            st.session_state["last_result"] = {
                "seed_text": seed_text,
                "num_words": num_words,
            }

    st.divider()

    result = st.session_state.get("last_result")

    if result is None:
        st.info(
            "👋 Enter a prompt above and click **Predict Next Word** to compare the two models."
        )
    else:
        st.subheader("Input Sentence")
        st.text_input(
            "Input sentence",
            value=result["seed_text"],
            disabled=True,
        )

        st.subheader("📊 Results")
        col_lstm, col_gru = st.columns(2, gap="large")

        lstm_word = render_model_card(
            col_lstm,
            "LSTM",
            "🔵",
            lstm_model,
            tokenizer,
            result["seed_text"],
            input_length,
            result["num_words"],
        )
        gru_word = render_model_card(
            col_gru,
            "GRU",
            "🟢",
            gru_model,
            tokenizer,
            result["seed_text"],
            input_length,
            result["num_words"],
        )

        st.divider()
        render_comparison(lstm_word, gru_word)

    st.divider()
    st.caption("Developed by Anas Kazi")
    st.caption("TensorFlow • Keras • Streamlit")


if __name__ == "__main__":
    main()
