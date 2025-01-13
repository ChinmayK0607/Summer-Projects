# Facial Landmark Detection

##  **Table of Contents**
- [**FACIAL LANDMARK DETECTION**](#FACIAL-LANDMARK-DETECTION)
  - [**Table of Contents**](#table-of-contents)
  - [**Objective**](#OBJECTIVE)
  - [**Learning Process**](#LEARNING-PROCESS)
  - [**Custom Dataset**](#CUSTOM-DATASET)
  - [**Facial Landmark Detection**](#FACIAL-LANDMARK-DETECTION)
    - [Dataset](#Dataset)
    - [Procedure](#PROCEDURE)
      - [1. Augmentation](#1Augmentation)
      - [2. Preprocessor](#2Preprocessor)
      - [3. Dataset Class](#3Dataset-Class)
  - [**Network Design**](#Network-Design)
      - [Architecture](#Architecture)
      - [Enhancements](#Enhancements)
      - [Training Loop](#Training-Loop)
   - [**Hyperparameters**](#HYPERPARAMETERS)
  - [**Results**](#results)

## Objective
The goal of the project is to develop a highly efficient model capable of accurately detecting facial key points in input images. This model will support real-time inference for applications such as:
- Face recognition
- Emotion detection
- Virtual makeup
- Augmented reality (AR)

## Learning Process

1. **Deep Learning Basics**: 
    - Started with understanding Deep Learning, Artificial Neural Networks, and Numpy.
    - Developed a basic MNIST model.
2. **Loss Functions and Optimizers**: 
    - Studied loss functions like cross-entropy loss and optimizers such as Adam.
    - Built an MNIST model using PyTorch.
3. **CNN Architectures**: 
    - Explored Convolutional Neural Networks (CNNs).
    - Implemented a CIFAR-10 model.
4. **Custom Dataset**: Created a custom dataset for facial landmark detection , Download the dataset from  [https://www.kaggle.com/datasets/toxicloser/ibug-300w-large-face-landmark-dataset].
5. **Facial Landmark Algorithm**: Learned and implemented the facial landmark detection algorithm using the PyTorch framework.

---

## Custom Dataset

- **Source**: The dataset is sourced from the iBUG 300-W dataset, which includes XML-based annotations for facial landmarks and crop coordinates.
- **Pipeline**:
    - Extracts image paths, 68 landmark points, and face cropping coordinates.
    - Resizes images with padding and applies random augmentations such as:
        - Color jitter
        - Offset cropping
        - Random cropping
        - Random rotation
    - Adjusts landmarks to match the augmentations.
    - Normalizes landmarks relative to image dimensions.
    - Converts images to grayscale and transforms them into tensors normalized between [-1, 1].
- **Dataset Class**: Handles parsing, preprocessing, and returning ready-to-use images and normalized landmarks.

### Result

**Sample Output**:

![Output Image](https://hackmd.io/_uploads/S1GGjmYkkg.png)

---

## Facial Landmark Detection

### Dataset

- **Commonly Used Datasets**: 
    - iBUG 300-W: Thousands of images labeled with 68 facial landmarks (eyes, nose, mouth, jawline).
    - **Applications**: Face recognition, emotion analysis, augmented reality.

---

### Procedure

1. **Augmentation**:
    - Implemented a `FaceLandmarksAugmentation` class for tailored augmentation techniques:
        - Cropping
        - Random cropping
        - Random rotation
        - Color jittering
    - Key parameters include image dimensions, brightness, and rotation limits.
    - Methods like `offset_crop` and `random_rotation` adjust landmark coordinates accordingly.

2. **Preprocessor**:
    - Initializes augmentation methods.
    - Normalizes data.

3. **Dataset Class**:
    - Inherits from `Dataset`.
    - Handles image paths, landmark coordinates, and cropping information parsed from XML files.
    - Splits data into training and validation sets.
    - Utilizes `DataLoader` for batch processing.

**Visualization Example**:

![Visualization](https://hackmd.io/_uploads/HkrIMVtk1l.png)

---

### Network Design

- **Architecture**:
    - Modular CNN with depthwise separable convolutions to improve efficiency.
    - **Entry Block**: Initial feature extraction.
    - **Middle Blocks**: Residual connections.
    - **Exit Block**: Outputs facial landmark coordinates.
- **Enhancements**:
    - Batch normalization and LeakyReLU for better performance.

- **Training Loop**:
    - Runs for 30 epochs.
    - Computes training loss and performs backpropagation.
    - Updates weights using an optimizer.
    - Validates model after each epoch.
    - Saves checkpoints to prevent overfitting and retain the best model.

**Network Design Visualization**:

![Network Design](https://hackmd.io/_uploads/BJo3HEYJkl.png)

---

### Hyperparameters

| Hyperparameter    | Value       |
|-------------------|-------------|
| Batch Size        | 32          |
| Learning Rate     | 0.00075     |
| Number of Epochs  | 30          |
| Loss Function     | MSE Loss    |
| Optimizer         | Adam        |

---

### Results

#### Loss

**Loss at Epoch 1**:

![Loss Epoch 1](https://hackmd.io/_uploads/S1kkuNKJ1l.png)

**Loss at Epoch 2**:

![Loss Epoch 2](https://hackmd.io/_uploads/H1C-OEKykg.png)

**Loss at Epoch 3**:

![Loss Epoch 3](https://hackmd.io/_uploads/HJfr_4Kyyx.png)

#### Graphs

![Training Graphs](https://hackmd.io/_uploads/S1qu_VYkyg.png)

---

## Software Tools Used

- Python
- Numpy
- PyTorch
- PIL
- Matplotlib
