"""
config.py
---------
Configuration settings and constants for the Image Caption Generator.
"""

import os

# Base paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# Model files
MODEL_PATH = os.path.join(MODELS_DIR, "image_caption_model.keras")
TOKENIZER_PATH = os.path.join(MODELS_DIR, "tokenizer.pkl")

# Check if model files exist
MODELS_READY = os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH)

# Hyperparameters
MAX_LENGTH = 34  # Maximum length of caption sequence
IMAGE_SIZE = (299, 299)  # Expected image size for InceptionV3
FEATURE_DIM = 2048  # Dimension of InceptionV3 feature vector

# Demo settings
DEMO_CAPTIONS = [
    "A dog running through an open grassy field on a sunny day.",
    "A group of people walking along a busy city street.",
    "A cat sitting on a windowsill looking outside.",
    "A young child playing with a ball in the park.",
    "A man riding a bicycle along a scenic coastal road.",
    "A woman standing in front of a beautiful mountain landscape.",
    "Two people sitting at a table and sharing a meal.",
]
