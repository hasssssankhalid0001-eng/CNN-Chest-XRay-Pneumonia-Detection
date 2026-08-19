# Chest X Ray Pneumonia Detection Using Deep Learning

This project focuses on developing a deep learning based system for detecting pneumonia from chest X ray images.

The project was developed as both a research experiment and an end to end machine learning application. Three different modelling approaches were investigated and compared: a custom CNN trained from scratch, MobileNetV2 transfer learning, and fine tuned MobileNetV2.

The final MobileNetV2 transfer learning model was integrated with a FastAPI backend and a Streamlit web application and deployed on Render.

## Live Application

Streamlit Application

https://cnn-chest-xray-streamlit.onrender.com/

FastAPI Backend

https://cnn-chest-xray-pneumonia-detection.onrender.com/

FastAPI Swagger Documentation

https://cnn-chest-xray-pneumonia-detection.onrender.com/docs

## Research Objective

The primary objective of this project was to investigate different deep learning approaches for binary classification of chest X ray images into NORMAL and PNEUMONIA classes.

The experiments were designed to answer three main questions:

1. How well can a CNN trained from scratch classify chest X ray images?

2. Does transfer learning using a pretrained MobileNetV2 improve generalization?

3. Does fine tuning the pretrained MobileNetV2 further improve performance on unseen data?

The models were evaluated using accuracy, precision, recall and F1 score. Particular attention was given to pneumonia recall because false negative pneumonia predictions are important when evaluating the effectiveness of a pneumonia classification system.

## Dataset

The project uses the Chest X Ray Pneumonia dataset containing two classes:

NORMAL

PNEUMONIA

The original directory structure contained approximately:

| Dataset | Images |
| --- | ---: |
| Training | 5,216 |
| Validation | 16 |
| Testing | 624 |

For model development, the original training data was reorganized into training and validation subsets.

The resulting dataset contained approximately:

| Dataset | Images |
| --- | ---: |
| Training | 4,694 |
| Validation | 522 |
| Testing | 624 |

The test set was kept separate and was used only for final evaluation.

## Image Preprocessing

The X ray images were resized to:

```text
224 × 224 pixels
