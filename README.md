# Create the README.md file with the content

readme_content = """
# Convolutional Neural Network (CNN) for CIFAR-100 Classification

This repository contains a Jupyter Notebook that demonstrates how to build and train a Convolutional Neural Network (CNN) to classify images in the CIFAR-100 dataset. The notebook includes steps for data preprocessing, model architecture definition, training, evaluation, and model saving.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Conclusion](#conclusion)
- [Usage](#usage)

## Introduction

This project demonstrates the implementation of a CNN for image classification on the CIFAR-100 dataset. The goal is to achieve high accuracy by experimenting with various model architectures and hyperparameters.

## Dataset

The CIFAR-100 dataset consists of 100 classes with 600 images each. There are 500 training images and 100 testing images per class. The images are 32x32 pixels with RGB color channels.

## Model Architecture

The CNN model is defined with the following architecture:

- Convolutional layers with ReLU activation
- Batch normalization layers
- Max pooling layers
- Dropout layers
- Fully connected layers

## Training

Training is performed using the Adam optimizer and Cross-Entropy loss function.  A hyperparam search code also utilizied to find best parameters.

## Evaluation

The model is evaluated on the test set after training. The performance metrics include loss and accuracy.

## Conclusion

- **Increased Convolutional Filters**: Improved feature extraction with deeper layers.
- **Added Batch Normalization**: Enhanced training stability and convergence.
- **Modified Pooling Layers**: Better spatial feature selection with additional pooling.
- **Increased Dropout Regularization**: Reduced overfitting with added dropout layers.
- **Updated Fully Connected Layers**: Enhanced capacity and stability in dense layers.
- **Overall Performance**: Achieved 66.11% accuracy on the test set, indicating significant improvements in model performance.

Test set: Average loss: 1.6720, Accuracy: 6611/10000 (66.11%)



## Usage

To use this notebook, follow these steps:

1. Clone the repository:
   \`\`\`bash
   git clone https://github.com/yourusername/cnn-cifar100.git
   cd cnn-cifar100
   \`\`\`

2. Install the required dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. Open the notebook:
   \`\`\`bash
   jupyter notebook DA520_HW2_CNN.ipynb
   \`\`\`

4. Run the cells sequentially to preprocess the data, build and train the model, evaluate performance, and save the model.
