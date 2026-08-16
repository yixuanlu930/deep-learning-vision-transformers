# Deep Learning: Computer Vision & Transformers

A deep learning project combining **computer vision, transfer learning, fine-tuning, and Transformer-based time-series forecasting** using PyTorch.

The repository contains two main experiments:

1. **Road-arrow orientation estimation from images**
2. **Transformer-based sequence forecasting**

The objective is to compare different deep learning strategies across both visual and temporal data.

---

## Overview

This project explores two complementary areas of modern Deep Learning.

### Computer Vision

The first part focuses on estimating the **orientation angle of road arrows painted on pavement**.

The problem is formulated as an image-regression task rather than a standard classification problem.

Several approaches are compared:

* Custom CNN trained from scratch
* Transfer learning with frozen pretrained backbones
* Full fine-tuning of pretrained CNN architectures

The pretrained models evaluated include:

* MobileNetV2
* ResNet18
* VGG16

### Transformers

The second part explores Transformer architectures for numerical sequence prediction.

Two experiments are implemented:

* Forecasting a synthetic sine wave
* Forecasting Apple stock prices using historical AAPL data

---

# Part I — Road Arrow Orientation Estimation

## Problem

The objective is to estimate the direction of arrows painted on roads from cropped image patches.

Rather than predicting a bounding box orientation, the model directly predicts the angular orientation of the arrow.

Each sample contains:

```text
image path
+
normalized orientation angle
```

The target angle is represented as a value in:

```text
[0, 1]
```

where:

```text
angle_degrees = 360 × target
```

---

## Dataset

The repository includes:

```text
dataset_cleaned.csv
```

Each row follows the format:

```text
image_path<TAB>normalized_angle
```

Example:

```text
dataset/...png    0.7472562211
```

The image is resized to:

```text
64 × 64 pixels
```

before being passed to the model.

The dataset is divided into approximately:

```text
70% training
10% validation
20% test
```

---

# Circular Angle Representation

Angular prediction has a particular difficulty:

```text
0° ≈ 360°
```

A standard scalar regression model can incorrectly consider these values very far apart.

To avoid this issue, the model represents orientation using:

```text
(cos θ, sin θ)
```

The output is normalized using an L2 normalization layer so that the predicted vector lies approximately on the unit circle.

Conceptually:

```text
Image
  │
  ▼
CNN
  │
  ▼
2D output
  │
  ▼
L2 normalization
  │
  ▼
(cos θ, sin θ)
  │
  ▼
Predicted angle
```

This representation handles the circular nature of angles more naturally.

---

# Custom CNN

The first approach builds a convolutional neural network from scratch.

The network learns visual features directly from the arrow images without relying on pretrained weights.

The training pipeline includes:

* PyTorch tensors
* DataLoader batching
* Training and validation loops
* GPU acceleration when available
* Angle-aware evaluation

The device is selected automatically between:

```text
Apple MPS
CUDA
CPU
```

---

# Evaluation Metrics

Several metrics are used to evaluate angular predictions.

These include:

* Mean Squared Error
* Mean Absolute Error in degrees
* R² score
* Accuracy within ±5°
* Accuracy within ±15°

These tolerance-based metrics are particularly useful because small angular errors may still represent practically correct orientation estimates.

---

# Transfer Learning

The project evaluates several ImageNet-pretrained CNN models:

```text
MobileNetV2
ResNet18
VGG16
```

Two strategies are compared.

## Frozen Feature Extractor

The pretrained convolutional backbone is frozen.

Only the newly added regression head is trained.

This approach is computationally efficient but showed limited performance for this particular task.

The pretrained ImageNet features are not sufficiently specialized for fine-grained road-arrow orientation estimation when the backbone remains completely frozen.

---

# Full Fine-Tuning

The second transfer-learning strategy allows **all pretrained layers to update their weights**.

The entire network is fine-tuned using a lower learning rate.

This produced substantially better results.

Among the evaluated architectures, **VGG16 achieved the strongest performance**.

Reported results include approximately:

```text
Accuracy ±15° ≈ 93%
Accuracy ±5°  ≈ 57%
R²            ≈ 0.87
```

These results show that full fine-tuning is significantly more effective than keeping the ImageNet backbone frozen.

---

# Transfer Learning Comparison

The experiments compare three strategies:

| Approach                 | Characteristics                                |
| ------------------------ | ---------------------------------------------- |
| Custom CNN               | Learns all visual features from scratch        |
| Frozen Transfer Learning | Uses pretrained features with a trainable head |
| Full Fine-Tuning         | Adapts the complete pretrained network         |

The results demonstrate that pretrained models become significantly more useful when they are allowed to adapt to the specific visual domain.

---

# Part II — Transformer Sequence Forecasting

The second notebook explores Transformer architectures for numerical sequence prediction.

The implementation is based on:

```text
PyTorch Transformer Encoder
+
Sinusoidal positional encoding
+
Linear input projection
```

---

# Experiment 1 — Sine-Wave Forecasting

A synthetic sine wave is generated:

```python
sin(t)
```

using:

```text
1000 points
```

Sequential windows of:

```text
50 previous values
```

are used to predict the next value.

---

## Transformer Architecture

The model projects each scalar input into a higher-dimensional representation.

Example architecture:

```text
Sequence
   │
   ▼
Linear projection
   │
   ▼
Positional encoding
   │
   ▼
Transformer Encoder
   │
   ▼
Final timestep representation
   │
   ▼
Linear output
   │
   ▼
Next value
```

The model uses:

```text
d_model = 64
multi-head self-attention
```

and sinusoidal positional encodings.

---

## Autoregressive Forecasting

After training, the model predicts future points iteratively.

For each step:

```text
current sequence
      │
      ▼
Transformer
      │
      ▼
next prediction
      │
      ▼
append prediction
      │
      ▼
remove oldest value
      │
      ▼
repeat
```

A total of:

```text
200 future points
```

are generated.

---

## Sine-Wave Evaluation

The predictions are compared against the real future sine signal.

Metrics include:

* MAE
* MSE
* RMSE
* R²
* Maximum error
* Pearson correlation
* Amplitude degradation

The model successfully learns the periodic structure of the signal, although autoregressive error gradually accumulates during long-horizon forecasting.

---

# Experiment 2 — Apple Stock Forecasting

The second Transformer experiment uses historical Apple stock data.

Data is downloaded using:

```text
yfinance
```

with ticker:

```text
AAPL
```

and historical data from approximately:

```text
2020 → 2026
```

The model uses the closing price as the target variable.

---

## Data Preparation

Stock values are normalized using:

```text
MinMaxScaler
```

Sequences of:

```text
60 trading days
```

are used to predict the next value.

The architecture again uses a Transformer Encoder with positional information.

---

# Future Forecast

The trained model generates:

```text
50 future business-day predictions
```

using an autoregressive process.

The predicted prices are then transformed back to the original dollar scale.

---

# Apple Model Evaluation

The project evaluates the model on historical known data before generating future predictions.

Reported metrics include approximately:

```text
MAE  ≈ $2.40
RMSE ≈ $3.26
MSE  ≈ 10.62
```

Additional metrics include:

* R²
* sMAPE
* Maximum error

---

# Limitations of Financial Forecasting

The project explicitly discusses the limitations of applying Transformers to stock-market prediction.

Unlike the sine wave, financial markets are:

* Noisy
* Non-stationary
* Influenced by external events
* Sensitive to geopolitical decisions
* Sensitive to macroeconomic shocks
* Influenced by information that may not exist in historical price data

Therefore, good historical fit does not imply reliable future financial prediction.

The stock experiment is intended as a **Deep Learning time-series exercise**, not as a financial forecasting system.

---

# Project Structure

```text
deep-learning-vision-transformers/
│
├── dataset_cleaned.csv
│
├── flechas_alumno_grupo_3_Xianye_Junjing_Yixuan.ipynb
├── transformer_alumno_grupo_3_Xianye_Junjing_Yixuan.ipynb
└── README.md
```

---

# Notebooks

## Road Arrow Orientation

```text
flechas_alumno_grupo_3_Xianye_Junjing_Yixuan.ipynb
```

Contains:

* Dataset loading
* Image preprocessing
* Custom CNN
* Circular angle representation
* Training loops
* Transfer learning
* MobileNetV2
* ResNet18
* VGG16
* Frozen-backbone experiments
* Full fine-tuning
* Model comparison

---

## Transformers

```text
transformer_alumno_grupo_3_Xianye_Junjing_Yixuan.ipynb
```

Contains:

* Synthetic sine-wave generation
* Sequence creation
* Transformer architecture
* Positional encoding
* Sine-wave forecasting
* Forecast evaluation
* AAPL data retrieval
* Financial time-series preprocessing
* Transformer training
* 50-day autoregressive forecast
* Model evaluation

---

# Installation

A Python environment with Jupyter is recommended.

Install the main dependencies:

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn opencv-python yfinance jupyter
```

---

# Running the Project

Start Jupyter:

```bash
jupyter notebook
```

For the computer-vision experiments, open:

```text
flechas_alumno_grupo_3_Xianye_Junjing_Yixuan.ipynb
```

For Transformer experiments, open:

```text
transformer_alumno_grupo_3_Xianye_Junjing_Yixuan.ipynb
```

---

# Technologies

## Deep Learning

* PyTorch
* Torchvision
* CNNs
* Transformers
* Transfer Learning
* Fine-Tuning

## Computer Vision

* OpenCV
* MobileNetV2
* ResNet18
* VGG16
* Image regression

## Time Series

* Transformer Encoder
* Positional Encoding
* Autoregressive Forecasting
* MinMaxScaler
* yfinance

## Data Analysis

* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Jupyter Notebook

---

# Key Concepts

This project explores:

* Deep Learning
* Computer Vision
* Convolutional Neural Networks
* Transfer Learning
* Fine-Tuning
* Circular regression
* Angle estimation
* ImageNet pretrained models
* Transformers
* Multi-head attention
* Positional encoding
* Time-series forecasting
* Autoregressive prediction
* Model evaluation
* GPU acceleration

---

# Main Findings

The experiments highlight several important observations.

### Fine-tuning matters

Using frozen ImageNet feature extractors produced weak results for the arrow-orientation task.

Allowing the pretrained model to adapt to the new domain dramatically improved performance.

### VGG16 performed best

Among the tested pretrained CNNs, VGG16 achieved the strongest orientation-estimation results after full fine-tuning.

### Transformers work well on structured periodic signals

The sine-wave experiment shows that Transformer architectures can model predictable temporal patterns effectively.

### Financial time series are fundamentally harder

The AAPL experiment demonstrates that good historical metrics do not remove the uncertainty associated with real financial markets.

---

# Academic Context

This project was developed as part of a practical assignment in **Deep Learning and Generative Artificial Intelligence**.

The objective was to explore different neural architectures and training strategies across computer vision and sequential data.

---

# License

See the repository license for applicable terms.

