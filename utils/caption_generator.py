"""
caption_generator.py
--------------------
Handles loading the LSTM model and tokenizer, and generating captions.
"""

import pickle
import numpy as np
import tensorflow as tf
from keras.models import load_model
from utils.config import MODEL_PATH, TOKENIZER_PATH, MAX_LENGTH

def load_caption_model():
    """
    Loads the trained LSTM model.
    """
    return load_model(MODEL_PATH)

def load_tokenizer():
    """
    Loads the fitted tokenizer.
    """
    with open(TOKENIZER_PATH, "rb") as f:
        return pickle.load(f)

def _pad_pre(seq, maxlen):
    """Pre-pad a sequence to maxlen with zeros."""
    arr = np.zeros((1, maxlen), dtype="int32")
    if len(seq) > 0:
        n = min(len(seq), maxlen)
        arr[0, maxlen - n:] = seq[-n:]
    return arr

def generate_caption(lstm_model, tokenizer, features: np.ndarray) -> str:
    """
    Greedy-search decoding to generate a caption string.
    """
    idx2word = {i: w for w, i in tokenizer.word_index.items()}

    # Pre-convert feature to tensor once
    feat_tensor = tf.constant(features, dtype=tf.float32)

    in_text = "startseq"
    for _ in range(MAX_LENGTH):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq_arr = _pad_pre(seq, MAX_LENGTH)
        seq_tensor = tf.constant(seq_arr, dtype=tf.int32)

        # Direct call skips .predict() overhead
        pred = lstm_model([feat_tensor, seq_tensor], training=False).numpy()
        word = idx2word.get(int(np.argmax(pred)), None)
        
        if word is None or word.lower() == "endseq":
            break
        in_text += " " + word

    caption = (
        in_text
        .replace("startseq", "")
        .replace("endseq", "")
        .strip()
    )
    
    if caption:
        caption = caption[0].upper() + caption[1:]
        if not caption.endswith("."):
            caption += "."
            
    return caption or "Could not generate a caption for this image."
