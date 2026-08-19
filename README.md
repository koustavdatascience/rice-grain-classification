# Rice Grain Classification

> **College final-year project** for classifying a rice-grain image into one of five trained varieties.

| Project detail | Value |
|---|---|
| Framework | Python, Flask, Keras, and TensorFlow |
| Current model scope | Arborio, Basmati, Ipsala, Jasmine, and Karacadag |

## Overview

This project provides a small Flask web application where a user uploads a clear image of rice grains. The Keras model processes the image and returns a predicted variety with a confidence score. The interface is deliberately simple: a focused upload area, a transparent result card, and an explanation of the five varieties covered by the current model.

## Run Locally

Clone the repository, install the Python dependencies, make sure the trained model is available, and start Flask.

```bash
git clone https://github.com/koustavdatascience/rice-grain-classification.git
cd rice-grain-classification
git lfs install
git lfs pull
pip install -r requirements-local.txt
python app.py
```

Then open `http://127.0.0.1:5000` in a browser.

## Project Structure

| Path | Purpose |
|---|---|
| `app.py` | Flask application and prediction flow. |
| `templates/index.html` | User interface for image upload and prediction results. |
| `static/uploads/` | Uploaded-image storage for the local application. |
| `rice_classifier_model.keras` | Trained Keras model, tracked through Git LFS. |
| `requirements-local.txt` | Full local Python dependencies, including TensorFlow. |
| `requirements.txt` | Lightweight Vercel interface dependencies. |

## Notes

The current application is limited to the five varieties listed above. It is an academic project and should be treated as an indicative image-classification tool rather than a certified seed-identification system.

## Vercel Preview

The Vercel deployment serves the polished project interface. Live TensorFlow inference remains a local-only feature until a deployable model artifact and a suitable inference runtime are available. Local model use requires `requirements-local.txt` and `git lfs pull`.
