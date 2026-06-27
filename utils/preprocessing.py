"""
preprocessing.py
----------------
Image validation and preprocessing logic.
"""

from PIL import Image, UnidentifiedImageError
import numpy as np

def validate_and_load_image(uploaded_file) -> Image.Image:
    """
    Validates the uploaded file and loads it as a PIL Image.
    Raises ValueError if the image is invalid or corrupted.
    """
    if uploaded_file is None:
        raise ValueError("No file provided.")
        
    try:
        # Load image and convert to RGB
        image = Image.open(uploaded_file)
        image = image.convert("RGB")
        return image
    except UnidentifiedImageError:
        raise ValueError("The uploaded file is not a valid image.")
    except Exception as e:
        raise ValueError(f"Failed to process image: {str(e)}")

def prepare_image_for_model(image: Image.Image, target_size: tuple) -> np.ndarray:
    """
    Resizes and converts a PIL Image to a numpy array for InceptionV3.
    """
    try:
        img = image.resize(target_size)
        arr = np.array(img, dtype="float32")
        arr = np.expand_dims(arr, axis=0)
        return arr
    except Exception as e:
        raise ValueError(f"Error during image resizing/conversion: {str(e)}")
