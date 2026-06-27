"""
feature_extractor.py
--------------------
Handles loading the CNN base model and extracting image features.
"""

import numpy as np

def load_inception_v3():
    """
    Loads InceptionV3 without the top classification layer.
    """
    from keras.applications.inception_v3 import InceptionV3
    from keras.models import Model
    
    base = InceptionV3(weights="imagenet")
    # Output from the last hidden layer -> (batch, 2048)
    model = Model(inputs=base.inputs, outputs=base.layers[-2].output)
    return model

def extract_features(extractor, image_array: np.ndarray) -> np.ndarray:
    """
    Normalizes the image array and extracts features using the provided model.
    """
    from keras.applications.inception_v3 import preprocess_input
    
    arr = preprocess_input(image_array.copy())
    # Direct tensor call avoids .predict() overhead
    features = extractor(arr, training=False).numpy()
    return features
