# VisionCaption AI

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red.svg)

A professional, full-stack Image Captioning application that uses deep learning to generate human-like descriptions of uploaded images.

## Overview
This project implements an end-to-end AI pipeline for image caption generation. Users can upload an image (JPG/PNG), and the application instantly processes it using a modular backend architecture to predict a descriptive caption.

### Problem Statement
Generating captions for images automatically is a fundamental challenge in bridging computer vision and natural language processing. This project solves this by encoding visual features and decoding them into structured linguistic patterns.

## Dataset
Trained on the standard benchmark dataset for image captioning.
- **Dataset:** Flickr8k
- **Images:** 8,091
- **Captions:** 40,455 (5 per image)
- **Vocabulary:** 8,811 unique words

## Architecture
The application uses a CNN-RNN encoder-decoder architecture:
1. **Encoder (InceptionV3):** A pre-trained InceptionV3 model (trained on ImageNet) acts as the feature extractor, removing the top classification layer to output a dense 2048-dimensional feature vector.
2. **Decoder (LSTM):** A custom Long Short-Term Memory (LSTM) network takes the image features and sequence data to predict the next word in the caption.

## Workflow
1. **Frontend (Streamlit):** User uploads an image through the highly-polished, responsive UI.
2. **Preprocessing:** The image is validated, resized to 299x299, and converted into a normalized NumPy array.
3. **Feature Extraction:** InceptionV3 extracts the 2048-dim feature vector.
4. **Caption Generation:** The LSTM decoder performs greedy search, predicting words iteratively until `endseq` or the maximum length (34 words) is reached.
5. **Post Processing:** The raw string is cleaned, capitalized, and formatted into a proper sentence before being displayed to the user.

## Training Process
- **Epochs:** 20
- **Loss:** Decreased from 5.23 → 2.27
- **Hardware:** Trained utilizing GPU acceleration (Google Colab).

## Model Limitations & Quirks
Because this model is trained from scratch on a very specific, limited dataset (Flickr8k), it has strong domain limitations:
- **Missing World Knowledge:** The dataset consists exclusively of people, children, dogs, bicycles, and grassy fields. It contains *zero* pictures of the moon, space, or complex cities.
- **Forced Guessing:** When presented with an out-of-domain image (like the moon against a dark sky), the model is forced to guess based on shapes it knows. A bright circle in the sky might be interpreted as a frisbee or a person jumping, leading to "hallucinated" captions (e.g., *"The girl is flying through the air"*).
- **Proper Testing:** To see the model perform accurately, test it with images similar to its training data: dogs on grass, people riding bikes, or children playing.

## Tech Stack
- **Frontend:** Streamlit, Atomic Inline CSS (Tailwind aesthetic)
- **Backend/ML:** TensorFlow, Keras, NumPy, Pillow
- **Deployment:** Streamlit Cloud Ready

## Installation

1. Clone the repository.
2. Ensure you have Python 3.13 installed.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Place the required model files into the `models/` directory:
   - `image_caption_model.keras`
   - `tokenizer.pkl`

## Usage

Run the Streamlit server locally:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

## Future Improvements
- Implement Beam Search decoding for higher quality captions.
- Add Attention mechanisms (Bahdanau/Luong) to improve word-to-region context.
- Support for larger datasets like MS COCO.
- Evaluate model using BLEU metric scores.
