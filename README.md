# Chest X Ray Pneumonia Detection Using Deep Learning

This project focuses on developing a deep learning based system for detecting pneumonia from chest X ray images.

The project was developed as both a research experiment and an end to end machine learning application. Three different modelling approaches were investigated and compared: a custom CNN trained from scratch, MobileNetV2 transfer learning, and fine tuned MobileNetV2.

The final MobileNetV2 transfer learning model was integrated with a FastAPI backend and a Streamlit web application and deployed on Render.

## Live Application

### Streamlit Application

https://cnn-chest-xray-streamlit.onrender.com/

### FastAPI Backend

https://cnn-chest-xray-pneumonia-detection.onrender.com/

### FastAPI Swagger Documentation

https://cnn-chest-xray-pneumonia-detection.onrender.com/docs

## Research Objective

The primary objective of this project was to investigate different deep learning approaches for binary classification of chest X ray images into NORMAL and PNEUMONIA classes.

The experiments were designed to answer three main questions:

How well can a CNN trained from scratch classify chest X ray images?

Does transfer learning using a pretrained MobileNetV2 improve generalization?

Does fine tuning the pretrained MobileNetV2 further improve performance on unseen data?

The models were evaluated using accuracy, precision, recall and F1 score. Particular attention was given to pneumonia recall because false negative pneumonia predictions are important when evaluating the effectiveness of a pneumonia classification system.

## Dataset

The project uses the Chest X Ray Pneumonia dataset containing two classes:

NORMAL

PNEUMONIA

The original directory structure contained approximately:

| Dataset | Images |
|---|---:|
| Training | 5,216 |
| Validation | 16 |
| Testing | 624 |

For model development, the original training data was reorganized into training and validation subsets.

The resulting dataset contained approximately:

| Dataset | Images |
|---|---:|
| Training | 4,694 |
| Validation | 522 |
| Testing | 624 |

The test set was kept separate and was used only for final evaluation.

## Image Preprocessing

The X ray images were resized to:

224 × 224 pixels

The original grayscale images were converted to 3 channel images for compatibility with MobileNetV2.

The training dataset contained substantially more PNEUMONIA images than NORMAL images. Class weights were therefore calculated and incorporated during training to reduce the effect of class imbalance.

## Model Development

Three different models were developed and evaluated during the project.

### Model 1: Custom CNN

A CNN was developed from scratch using convolutional layers, batch normalization, max pooling, global average pooling, dropout and a dense output layer.

The purpose of this model was to establish a baseline and understand how a CNN trained from scratch would perform on the dataset.

The model achieved approximately:

| Metric | Result |
|---|---:|
| Test Accuracy | 62.5% |

However, analysis of the confusion matrix showed that the model predicted almost every test image as PNEUMONIA. This indicated poor class discrimination despite the model achieving a reasonable looking accuracy.

### Model 2: MobileNetV2 Transfer Learning

MobileNetV2 pretrained on ImageNet was used as a feature extractor with a custom binary classification head.

During the initial transfer learning stage, the pretrained MobileNetV2 layers were frozen and only the newly added classification layers were trained.

The model achieved the following results on the unseen test dataset:

| Metric | Result |
|---|---:|
| Test Accuracy | 86.38% |
| Precision | 83.81% |
| Recall | 96.92% |
| PNEUMONIA F1 Score | 0.90 |
| Macro F1 Score | 0.85 |
| Test Loss | 0.310 |

The model showed a substantial improvement over the custom CNN and demonstrated strong pneumonia detection capability.

### Model 3: Fine Tuned MobileNetV2

The upper layers of MobileNetV2 were then unfrozen and fine tuned using a very small learning rate.

Fine tuning produced extremely strong validation performance. However, the improvement observed during validation did not translate into better performance on the unseen test dataset.

The fine tuned model achieved approximately:

| Metric | Result |
|---|---:|
| Test Accuracy | 81.9% |
| PNEUMONIA Recall | 99.7% |

Although the model achieved a higher pneumonia recall, its overall test accuracy decreased and precision also decreased because of the increased number of false positive predictions.

## Model Comparison

The three approaches produced noticeably different results:

| Model | Test Accuracy | PNEUMONIA Recall | Overall Observation |
|---|---:|---:|---|
| Custom CNN | 62.5% | Poor class discrimination | Strong bias towards PNEUMONIA |
| MobileNetV2 Transfer Learning | 86.38% | 96.92% | Best overall balance |
| Fine Tuned MobileNetV2 | 81.9% | 99.7% | Highest recall but lower generalization |

The custom CNN provided a useful baseline but struggled to distinguish between the two classes.

Fine tuned MobileNetV2 achieved the highest pneumonia recall, reaching approximately 99.7%. However, this came at the cost of lower precision and lower overall test accuracy.

The original MobileNetV2 transfer learning model achieved the highest test accuracy of 86.38% while maintaining a high pneumonia recall of 96.92% and a PNEUMONIA F1 score of 0.90.

Therefore, the frozen MobileNetV2 transfer learning model was selected as the final model because it provided the best overall balance between accuracy, precision, recall and F1 score on the unseen test dataset.

## Threshold Analysis

Different classification thresholds were investigated instead of relying only on the default 0.5 threshold.

Lower thresholds increased pneumonia recall but also produced substantially more false positive predictions.

Higher thresholds reduced false positives but increased the possibility of missing pneumonia cases.

The default 0.5 threshold provided a better overall balance for the final model and was therefore selected for the deployed application.

## Final Model Performance

The final selected MobileNetV2 transfer learning model achieved:

Accuracy: 86.38%

Precision: 83.81%

Recall: 96.92%

PNEUMONIA F1 Score: 0.90

Macro F1 Score: 0.85

The confusion matrix on the 624 image test set was:

| | Predicted NORMAL | Predicted PNEUMONIA |
|---|---:|---:|
| Actual NORMAL | 161 | 73 |
| Actual PNEUMONIA | 12 | 378 |

The model correctly identified 378 out of 390 PNEUMONIA images, resulting in a pneumonia recall of approximately 96.92%.

## Approach

The overall project was developed in three stages.

First, different deep learning approaches were experimented with to understand the performance of a custom CNN, transfer learning and fine tuning.

Second, the best performing model on the unseen test dataset was selected and saved as a Keras model.

Finally, the trained model was integrated into a FastAPI REST API and connected to a Streamlit interface. The Streamlit application allows a user to upload a chest X ray image, which is sent to the FastAPI backend for prediction. The backend processes the image using the same preprocessing pipeline and returns the predicted class, pneumonia probability and confidence.

The final system was deployed on Render so that both the frontend and backend can be accessed through the internet.


## Project Outcome

The project demonstrated that transfer learning using MobileNetV2 provided significantly better generalization than a CNN trained from scratch on this dataset.

The comparison between transfer learning and fine tuning also showed that higher validation performance does not necessarily translate into better performance on unseen test data.

The final MobileNetV2 transfer learning model was therefore selected based on its overall test performance and its balance between pneumonia detection capability and false positive predictions.

This project combines model experimentation, performance analysis, REST API development, frontend development and cloud deployment into a complete end to end deep learning application.

## Disclaimer

This application is an educational AI project and is not intended to provide medical diagnosis or replace professional medical advice.
