
# animal-image-classifier

# Overview

This project demonstrates an image classification pipeline using TensorFlow, Keras, and KerasCV. The model is trained on a custom dataset stored in Google Drive and utilizes transfer learning with a ResNet50V2 architecture.

# Features

  1. Provides an easy-to-use Gradio API for inference.

  2. Uses keras-cv for image classification.

  3. Implements a dataset pipeline using tf.data.

  4. Applies image normalization and augmentation techniques.

  5. Fine-tunes a pre-trained ResNet50V2 model.

  6. Implements callbacks like early stopping and model checkpointing.

  7. Evaluates model performance with accuracy and loss plots.
  
  8. Predicts on new images and saves model weights.

# Installation

Before running the project, install the required dependencies:

!pip install -q --upgrade keras-cv keras tensorflow tensorflow-datasets

# Dataset Preparation

The dataset is stored in Google Drive and loaded using TensorFlow's image_dataset_from_directory function. The dataset is split into training and validation sets with an 80-20 ratio.

# Model Training

  1. A pre-trained ResNet50V2 model is used.

  2. The model is compiled with AdamW optimizer and sparse_categorical_crossentropy loss.

  3. Training runs for 100 epochs with early stopping.

  4. The best model weights are saved using ModelCheckpoint.

# Evaluation

  1. The model is evaluated on the validation dataset.

  2. Training history plots show accuracy and loss trends.

# Prediction

The model predicts the class of an input image.
The predicted class label is displayed.
Saving Model Weights
model.save_weights('/content/drive/MyDrive/Internship/anim_class_model.weights.h5')

# Usage

  1. Mount Google Drive and set the dataset path.

  2. Train the model.

  3. Evaluate and visualize the results.

  4. Perform predictions on new images.

  5. Use the Gradio API for easy inference.

  6. Mount Google Drive and set the dataset path.

  7. Train the model.

  8. Evaluate and visualize the results.

  9. Perform predictions on new images.

# Author

This project is part of a Machine Learning Internship at Unified Mentor.
