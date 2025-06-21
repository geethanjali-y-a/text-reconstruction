🧠 Scene Text Detection and Next Word Prediction

 📌 Project Overview

This project presents a Deep Learning-based solution to recognize and predict missing or next words in scene text images — especially those with noise, tears, or incomplete text. The model integrates **OCR**, **image preprocessing**, and **NLP with LSTM** to handle real-world noisy text images and accurately infer what text is likely missing.

 🔍 Problem Statement

Scene text from real-world images (e.g., street signs, torn posters, old documents) often contains:
- Noisy backgrounds
- Partially visible or torn characters
- Incomplete words

This makes it difficult to read or process the content using conventional OCR systems.

 🎯 Solution Highlights

- 🧹 Image Preprocessing:
  - Noise removal, resizing, grayscale conversion, and thresholding.
  
- 🔎 Scene Text Detection:
  - Applied OCR (Optical Character Recognition) techniques for extracting readable text from images.

- 🧠 Next Word Prediction:
  - Used **LSTM (Long Short-Term Memory)** based NLP model trained on relevant corpus to predict the next likely word.
  - Integrated GPT-like transformer models for advanced completion (optional).

- 🔄 Robust Text Completion:
  - Predicts missing or torn text using context-based language modeling.

---

 🧰 Tech Stack

| Component           | Technology Used                    |
|---------------------|------------------------------------|
| Image Preprocessing | OpenCV, NumPy                      |
| OCR Engine          | Tesseract OCR                      |
| NLP & Prediction    | TensorFlow/Keras (LSTM), NLTK      |
| Transformer Model   | OpenAI GPT APIs or HuggingFace     |
| Backend (Optional)  | Flask / Streamlit (for deployment) |
| Programming Language| Python 3.x                         |

---

📁 Folder Structure

```

scene-text-predictor/
├── images/                    # Sample input images
├── preprocessing/             # Image cleanup scripts
├── ocr/                       # Text detection logic (Tesseract)
├── nlp\_model/                 # LSTM or transformer-based prediction model
├── app.py                     # Main app (Flask/Streamlit)
├── utils.py                   # Helper functions
├── README.md
└── requirements.txt

````

---

 🧪 Sample Workflow

1. Upload or pass a torn scene image
2. OCR detects readable text: `"Schoo"`
3. LSTM model predicts next word: `"School"`
4. Output: `"School"` is displayed as completion

---

📈 Model Training

* LSTM model trained on a custom dataset containing:

  * Academic text
  * Real-world phrases
  * Scene label data

* Tokenization, embedding, and sequence padding techniques were applied for accuracy.

