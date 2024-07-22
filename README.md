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

-     class ConvNetDA520_20(nn.Module):
        def __init__(self):
        # Initial image was 32 X 32 X 3
        super(ConvNetDA520_20, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, 3, padding=1) # 32 X 32 X 256
        self.bn1 = nn.BatchNorm2d(256) # 32 X 32 X 256
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1) # 32 X 32 X 256
        self.bn2 = nn.BatchNorm2d(256) # 32 X 32 X 256
        self.pool1 = nn.MaxPool2d(2, 2) # 16 X 16 X 256
        self.dropout1 = nn.Dropout(0.2) # 16 X 16 X 256

        self.conv3 = nn.Conv2d(256, 512, 3, padding=1) # 16 X 16 X 512
        self.bn3 = nn.BatchNorm2d(512) # 16 X 16 X 512
        self.conv4 = nn.Conv2d(512, 512, 3, padding=1) # 16 X 16 X 512
        self.bn4 = nn.BatchNorm2d(512) # 16 X 16 X 512
        self.pool2 = nn.MaxPool2d(2, 2) # 8 X 8 X 512
        self.dropout2 = nn.Dropout(0.2) # 8 X 8 X 512

        self.conv5 = nn.Conv2d(512, 512, 3, padding=1) # 8 X 8 X 512
        self.bn5 = nn.BatchNorm2d(512) # 8 X 8 X 512
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1) # 8 X 8 X 512
        self.bn6 = nn.BatchNorm2d(512) # 8 X 8 X 512
        self.pool3 = nn.MaxPool2d(2, 2) # 4 X 4 X 512
        self.dropout3 = nn.Dropout(0.2) # 4 X 4 X 512

        self.conv7 = nn.Conv2d(512, 512, 3, padding=1) # 4 X 4 X 512
        self.bn7 = nn.BatchNorm2d(512) # 4 X 4 X 512
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1) # 4 X 4 X 512
        self.bn8 = nn.BatchNorm2d(512) # 4 X 4 X 512
        self.pool4 = nn.MaxPool2d(2, 2) # 2 X 2 X 512
        self.dropout4 = nn.Dropout(0.2) # 2 X 2 X 512

        self.flatten = nn.Flatten() # 1 X 2048
        self.fc1 = nn.Linear(512 * 2 * 2, 1024) # 1 X 1024
        self.dropout5 = nn.Dropout(0.2) # 1 X 1024
        self.bn9 = nn.BatchNorm1d(1024, momentum=0.95, eps=0.005) # 1 X 1024
        self.fc2 = nn.Linear(1024, 100) # 1 X 1024

        def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool4(x)
        x = self.dropout4(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout5(x)
        x = self.bn9(x)
        x = self.fc2(x)
        return x

## Training

Training is performed using the Adam-SGD-RMSProp and Cross-Entropy loss function.  A hyperparam search code also utilizied to find best parameters.

## Evaluation

The model is evaluated on the test set after training. The performance metrics include loss and accuracy. The train-test loss graph for the final network is below.

![image](https://github.com/user-attachments/assets/79f0e86d-458a-497b-b98c-2d62ddd306aa)


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
   git clone https://github.com/fozgur/cnn-cifar100.git
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
