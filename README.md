# 🌳 Tree Species Identification Project

This repository contains a deep learning project to classify tree species based on leaf images using transfer learning techniques. The dataset is sourced from Kaggle and organized into weekly progress.

## 📁 Dataset Used

- **Source**: [Tree Species Identification Dataset – Kaggle](https://www.kaggle.com/datasets/viditgandhi/tree-species-identification-dataset)
- The dataset contains images of leaves categorized into folders by tree species.

## 🔧 Week 1 Activities Completed

- ✅ Downloaded the dataset from Kaggle
- ✅ Mounted Google Drive in Google Colab using:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')

## 📚 Week 2 Activities Completed
This week focused on building and training a tree species classifier using transfer learning with MobileNetV2.

### 🔄 Data Preprocessing
Used ImageDataGenerator to:

- Normalize pixel values with rescale=1./255

- Apply real-time augmentations (zoom, rotation, flipping)

- Loaded data using flow_from_directory

- Created separate training and validation generators

### 🧠 Model Architecture
Loaded MobileNetV2 with pre-trained ImageNet weights (include_top=False)

Added custom layers:

- GlobalAveragePooling2D

- Dense(128, activation='relu')

- Dropout(0.5)

- Final softmax Dense layer for classification

- Base model layers set to non-trainable to preserve learned features

### ⚙️ Compilation & Training
Compiled with:

- Optimizer: Adam

- Loss: categorical_crossentropy

- Metric: accuracy

- Trained the model over multiple epochs

- Plotted training and validation accuracy/loss

### 💾 Model Saving
Saved the trained model using:

python
Copy
Edit
model.save('tree_species_model.h5')

### 📊 Results
- Achieved promising initial accuracy

- Noted early signs of overfitting — to be addressed in Week 3 with tuning and regularization

