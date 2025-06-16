---
title: III - MNIST and Fashion-MNIST Classification
weight: 25
---

# MNIST and Fashion-MNIST Classification

## Introduction

In this lab session, we explore handwritten digit recognition using the MNIST dataset, progressing from classical machine learning approaches to modern deep learning techniques. We then apply transfer learning to the more challenging Fashion-MNIST dataset. This progression mirrors the historical development of the field while providing hands-on experience with key optimization concepts.

The session is divided into three parts:
- **Part I**: Classical approaches using Multi-Layer Perceptrons (MLP) and Support Vector Machines (SVM)
- **Part II**: Deep learning with Convolutional Neural Networks (CNN)
- **Part III**: Transfer learning applied to Fashion-MNIST

> **Note:** For this lab you will need a running Python with packages: `numpy`, `matplotlib`, `scikit-learn`, `pandase`, `torch`, `torchvision`.


## Learning objectives

By the end of this session, you should be able to:
- Implement backpropagation and stochastic gradient descent from scratch
- Design proper validation strategies to avoid overfitting
- Build and train CNNs using PyTorch
- Apply transfer learning to improve model performance
- Understand the trade-offs between different optimization strategies

> You need to implement a backpropagation procedure for MLP. Since we didn't have time to cover this, we refer to [this ressource](https://visionbook.mit.edu/backpropagation.html).

## Part I: Classical Machine Learning Approaches

### 1. Introduction and Data Exploration

The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28×28 pixels. We begin by loading and exploring this dataset. This initial setup is provided for you.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load MNIST data
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')

# Convert to appropriate types
X = X.astype(np.float32)
y = y.astype(np.int64)

print(f"Data shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Pixel value range: [{X.min()}, {X.max()}]")

# Visualize some examples
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X[i].reshape(28, 28), cmap='gray')
    ax.set_title(f'Label: {y[i]}')
    ax.axis('off')
plt.tight_layout()
plt.show()
```

For classical machine learning approaches, we work with flattened vectors rather than 2D images. Each image becomes a 784-dimensional vector where spatial structure is implicit in the feature ordering.

### 2. Data Preprocessing and Visualization

Preprocessing is crucial for optimization convergence. We'll normalize the data and use dimensionality reduction for visualization.

#### Normalization
Normalization ensures that all features contribute equally to the optimization process. For pixel values in range [0, 255], we have two main approaches:
1.  **Min-Max scaling**: $x\_{\text{scaled}} = \frac{x - x\_{\min}}{x\_{\max} - x\_{\min}} = \frac{x}{255}$
2.  **Standardization**: $x\_{\text{std}} = \frac{x - \mu}{\sigma}$

where $\mu$ is the mean and $\sigma$ is the standard deviation across the training set. We will use Min-Max scaling. The three-way data split (train, validation, test) is also performed here.

```python
from sklearn.preprocessing import MinMaxScaler

# We'll use MinMaxScaler for pixel data as it preserves the 0 boundary
scaler = MinMaxScaler()

# Create a three-way split for robust evaluation
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Fit scaler on training data only to avoid data leakage
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train_scaled.shape}")
print(f"Validation set: {X_val_scaled.shape}")
print(f"Test set: {X_test_scaled.shape}")
```

#### Dimensionality Reduction for Visualization
To understand the data structure, we apply PCA and t-SNE. PCA finds the directions of maximum variance through eigenvalue decomposition of the covariance matrix: $\mathbf{C} = \frac{1}{n-1}\mathbf{X}^T\mathbf{X}$. The following code will help you visualize the high-dimensional data in 2D.

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Apply PCA for visualization
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train_scaled)

# t-SNE for 2D visualization (on a subset for speed)
subset_size = 5000
indices = np.random.choice(len(X_train_scaled), subset_size, replace=False)
X_subset = X_train_scaled[indices]
y_subset = y_train[indices]

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_subset)

# Visualize t-SNE embedding
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_subset, cmap='tab10', alpha=0.6)
plt.colorbar(scatter)
plt.title('t-SNE visualization of MNIST digits')
plt.xlabel('t-SNE component 1')
plt.ylabel('t-SNE component 2')
plt.show()
```

### 3. Experimental Methodology
A rigorous experimental protocol is key to reliable results.

#### Three-way Data Splitting
We use a three-way split:
- **Training set (49,000 samples)**: For model parameter learning.
- **Validation set (10,500 samples)**: For hyperparameter tuning and early stopping.
- **Test set (10,500 samples)**: For final, unbiased model evaluation.

#### K-Fold Cross-Validation
For hyperparameter optimization, we employ K-fold cross-validation on the training set. The cross-validation error is:
$$\text{CV}(k) = \frac{1}{k} \sum\_{i=1}^{k} L(\mathbf{w}\_{-i}, \mathcal{D}_i)$$
where $\mathbf{w}\_{-i}$ is the model trained on all folds except fold $i$.

### 4. Multi-Layer Perceptron from Scratch
Now for the main challenge: implementing a two-layer neural network from scratch to understand backpropagation and stochastic optimization.

#### Network Architecture
- Input layer: 784 neurons
- Hidden layer: 128 neurons with ReLU activation
- Output layer: 10 neurons with softmax activation

The forward propagation equations are:
\begin{align}
\mathbf{z}^{(1)} &= \mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)} \\\\
\mathbf{a}^{(1)} &= \text{ReLU}(\mathbf{z}^{(1)}) \\\\
\mathbf{z}^{(2)} &= \mathbf{W}^{(2)}\mathbf{a}^{(1)} + \mathbf{b}^{(2)} \\\\
\hat{\mathbf{y}} &= \text{softmax}(\mathbf{z}^{(2)})
\end{align}

#### Loss Function
We use the cross-entropy loss:
$$L(\mathbf{W}, \mathbf{b}) = -\frac{1}{n}\sum\_{i=1}^{n}\sum\_{j=1}^{10} y\_{ij}\log(\hat{y}\_{ij})$$

#### Backpropagation Derivation
Here are the crucial gradients you'll need for implementation:
1.  **Output layer gradients**: $\frac{\partial L}{\partial \mathbf{z}^{(2)}} = \hat{\mathbf{y}} - \mathbf{y}$
2.  **Hidden layer gradients**: $\frac{\partial L}{\partial \mathbf{z}^{(1)}} = (\mathbf{W}^{(2)})^T \frac{\partial L}{\partial \mathbf{z}^{(2)}} \odot \mathbf{1}[\mathbf{z}^{(1)} > 0]$
3.  **Weight gradients**:
    $\frac{\partial L}{\partial \mathbf{W}^{(2)}} = \frac{1}{n}\sum\_{i=1}^{n} \frac{\partial L}{\partial \mathbf{z}^{(2)}_i} (\mathbf{a}^{(1)}_i)^T$
    and
    $\frac{\partial L}{\partial \mathbf{W}^{(1)}} = \frac{1}{n}\sum\_{i=1}^{n} \frac{\partial L}{\partial \mathbf{z}^{(1)}_i} \mathbf{x}_i^T$

#### Implementation

**Task: Implement the `MLPFromScratch` Class**

Your task is to create a Python class `MLPFromScratch` that builds and trains our two-layer neural network. Use the information to guide you through implementing each method.

1. __init__ & Helpers
**`__init__(self, ...)`**
-   Initialize weights and biases for both layers. Use Xavier initialization for weights (e.g., `np.random.randn(...) * np.sqrt(2.0 / n_input)`) to aid convergence. Initialize biases to zero.
-   Initialize velocity terms for momentum-based gradient descent (e.g., `self.vW1`, `self.vb1`) as zero arrays with the same shape as the corresponding parameters.

**Helper Functions**
-   `relu(self, x)`: Implement the ReLU activation function, $\max(0, x)$.
-   `relu_derivative(self, x)`: Return 1 for positive inputs, 0 otherwise.
-   `softmax(self, x)`: Implement the softmax function. Remember to subtract the max value from `x` before exponentiating for numerical stability: $\text{softmax}(\mathbf{z})_i = \frac{\exp(z_i - \max(\mathbf{z}))}{\sum_j \exp(z_j - \max(\mathbf{z}))}$.

2. Forward Pass
**`forward(self, X)`**
-   Implement the forward pass using the equations above.
-   The input `X` will have shape `(n_samples, n_features)`. It's often easier to work with column vectors, so you might need to transpose it.
-   Store intermediate values like `self.z1`, `self.a1`, `self.z2`, and `self.a2` as they are needed for backpropagation.
-   The method should return the final predictions, `a2`, transposed back to shape `(n_samples, n_classes)`.

**`compute_loss(self, y_pred, y_true)`**
-   Implement the cross-entropy loss.
-   You will need to convert the true labels `y_true` (e.g., `[5, 0, 4, ...]`) into one-hot encoded vectors.
-   Add a small epsilon (e.g., `1e-8`) to `y_pred` before taking the logarithm to avoid `log(0)`.

3. Backward Pass
**`backward(self, X, y_true, momentum=0.9)`**
-   This is the core of the learning process. Implement the backpropagation algorithm using the gradient equations provided.
-   Calculate the gradients for `W2`, `b2`, `W1`, and `b1`.
-   Update the velocity terms for each parameter using the momentum formula: $v_t = \text{momentum} \cdot v\_{t-1} - \text{lr} \cdot \nabla L$.
-   Update the weights and biases using their corresponding velocity terms: $W \leftarrow W + v_W$.

4. Train & Predict
**`train(self, X_train, y_train, ...)`**
-   Implement the main training loop.
-   Iterate for a given number of `epochs`.
-   In each epoch, shuffle the training data to ensure batches are random.
-   Implement mini-batch gradient descent: loop through the training data in batches of a specified `batch_size`.
-   For each batch, perform a `forward` pass, `compute_loss`, and a `backward` pass.
-   After each epoch, calculate and store the training loss and the accuracy on the validation set. This allows you to monitor for overfitting.

**`predict(self, X)`**
-   Perform a forward pass and return the index of the highest-scoring class for each input sample using `np.argmax`.

#### Training the MLP
Once your class is implemented, use the following code to train it and visualize the results. Experiment with different learning rates.

```python
# NOTE: This code assumes you have created the MLPFromScratch class.
# mlp = MLPFromScratch(learning_rate=0.01)
# train_losses, val_accuracies = mlp.train(
#     X_train_scaled, y_train,
#     X_val_scaled, y_val,
#     epochs=50, batch_size=128
# )

# Plot training curves
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
# ax1.plot(train_losses)
# ax1.set_xlabel('Epoch'); ax1.set_ylabel('Training Loss'); ax1.set_title('Training Loss over Epochs')
# ax2.plot(val_accuracies)
# ax2.set_xlabel('Epoch'); ax2.set_ylabel('Validation Accuracy'); ax2.set_title('Validation Accuracy over Epochs')
# plt.tight_layout(); plt.show()
```

Now, experiment with different batch sizes. How does batch size affect:
- Convergence speed?
- Final accuracy?
- Computational efficiency?

{{< hiddenhint "Hint" >}}
Smaller batch sizes (e.g., 32) introduce more noise into gradient estimates, which can help escape poor local minima but might make convergence erratic. Larger batch sizes (e.g., 256, 512) provide more stable gradients and faster computation per epoch but risk converging to sharper, less generalizable minima.
{{< /hiddenhint >}}

### 5. Support Vector Machines
Now we apply SVMs to the same problem using scikit-learn's optimized implementation. This serves as a powerful baseline to compare against our MLP. We use `GridSearchCV` to find the best hyperparameters.

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time

svm_model = SVC(kernel='rbf', random_state=42)
param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto', 0.01]}

# Use a subset for faster grid search
X_train_subset = X_train_scaled[:5000]
y_train_subset = y_train[:5000]

print("Starting Grid Search for SVM...")
start_time = time.time()
grid_search = GridSearchCV(svm_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train_subset, y_train_subset)
print(f"Grid search completed in {time.time() - start_time:.2f} seconds")

best_svm = grid_search.best_estimator_
best_svm.fit(X_train_scaled, y_train)
val_acc_svm = best_svm.score(X_val_scaled, y_val)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Validation accuracy: {val_acc_svm:.4f}")
```

### 6. Comparative Analysis
Let's compare the performance of your custom MLP with the tuned SVM on the final test set.

**Task: Create an sklearn-compatible Estimator**

To easily compare your MLP with sklearn models, create a wrapper class `MLPClassifier` that inherits from `sklearn.base.BaseEstimator` and `sklearn.base.ClassifierMixin`.

- The `__init__` method should store hyperparameters like `learning_rate`, `epochs`, etc.
- The `fit(self, X, y)` method should initialize and train your `MLPFromScratch` instance.
- The `predict(self, X)` method should call the `predict` method of your trained MLP.
- The `score(self, X, y)` method should predict on `X` and return the accuracy against `y`.

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# NOTE: This assumes you have created the MLPClassifier wrapper.
# mlp_sklearn = MLPClassifier(hidden_size=128, learning_rate=0.01, epochs=30)
# mlp_sklearn.fit(X_train_scaled, y_train)

# Predictions on the test set
# mlp_pred = mlp_sklearn.predict(X_test_scaled)
# svm_pred = best_svm.predict(X_test_scaled)
# ...
```
Analyze the computational costs.
{{< hiddenhint "Hint" >}}
Compare training time, prediction time, and memory usage. SVMs with RBF kernels have $O(n^2)$ training complexity, making them slow for large datasets. Neural networks have $O(n \cdot d \cdot h)$ complexity per epoch, which is more scalable.
{{< /hiddenhint >}}

## Part II: Deep Learning with CNNs
### 1. Transition to PyTorch
We now move to PyTorch for implementing Convolutional Neural Networks (CNNs). PyTorch provides automatic differentiation and GPU acceleration, which are essential for deep learning. The following boilerplate code prepares the data for PyTorch.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Convert numpy arrays to PyTorch tensors and reshape for CNNs (N, C, H, W)
X_train_tensor = torch.FloatTensor(X_train_scaled.reshape(-1, 1, 28, 28))
y_train_tensor = torch.LongTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val_scaled.reshape(-1, 1, 28, 28))
y_val_tensor = torch.LongTensor(y_val)
X_test_tensor = torch.FloatTensor(X_test_scaled.reshape(-1, 1, 28, 28))
y_test_tensor = torch.LongTensor(y_test)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

### 2. Convolutional Neural Network Design
CNNs exploit the spatial structure of images through local connectivity and parameter sharing.

#### CNN Architecture Implementation

**Task: Implement a Simple CNN in PyTorch**

Create a `SimpleCNN` class that inherits from `nn.Module`.

1.  **`__init__(self)`**:
    -   Define the layers of the network. Use the following architecture:
        1.  `nn.Conv2d`: 1 input channel, 32 output channels, kernel size of 3, padding of 1.
        2.  `nn.MaxPool2d`: kernel size of 2, stride of 2. (This will reduce 28x28 to 14x14).
        3.  `nn.Conv2d`: 32 input channels, 64 output channels, kernel size of 3, padding of 1.
        4.  `nn.MaxPool2d`: kernel size of 2, stride of 2. (This will reduce 14x14 to 7x7).
        5.  `nn.Linear`: Input size will be the flattened output of the previous layer (`64 * 7 * 7`). Output size of 128.
        6.  `nn.Dropout`: with a probability of 0.5 for regularization.
        7.  `nn.Linear`: 128 inputs, 10 outputs (for the 10 digit classes).

2.  **`forward(self, x)`**:
    -   Define the data flow through the network.
    -   Pass the input `x` through `conv1`, then apply a `F.relu` activation, then pass through the first `pool` layer.
    -   Repeat for the second convolutional block (`conv2`, `relu`, `pool`).
    -   Before the fully connected layers, you must flatten the tensor. Use `x = x.view(-1, 64 * 7 * 7)`.
    -   Pass the flattened tensor through `fc1`, `relu`, `dropout`, and finally `fc2`.
    -   Return the raw output scores (logits) from `fc2`. The loss function will handle the softmax.

### 3. Training Methodology
#### Training Loop Implementation

**Task: Implement a PyTorch Training Loop**

Write a function `train_model(model, train_loader, val_loader, ...)` that trains your `SimpleCNN`.

1.  **Setup**:
    -   Move the `model` to the selected `device`.
    -   Define the `criterion` (loss function): `nn.CrossEntropyLoss()`.
    -   Define the `optimizer`: `optim.Adam(model.parameters(), lr=...)`.

2.  **Outer Loop**: Iterate through `epochs`.

3.  **Training Phase (Inner Loop)**:
    -   Set the model to training mode: `model.train()`.
    -   Iterate through the `train_loader` to get batches of `data` and `target`.
    -   Move `data` and `target` to the `device`.
    -   **Crucially, zero the gradients**: `optimizer.zero_grad()`.
    -   Perform a forward pass: `output = model(data)`.
    -   Calculate the loss: `loss = criterion(output, target)`.
    -   Perform backpropagation: `loss.backward()`.
    -   Update the model weights: `optimizer.step()`.
    -   Keep track of running loss and accuracy.

4.  **Validation Phase (Inner Loop)**:
    -   Set the model to evaluation mode: `model.eval()`.
    -   **Disable gradient calculation** with `with torch.no_grad():`.
    -   Iterate through the `val_loader`.
    -   Calculate the validation loss and accuracy for the epoch.

5.  **Logging & Return**:
    -   Print the training and validation statistics at the end of each epoch.
    -   Return a history dictionary containing lists of training losses, validation losses, etc., for later plotting.

### 4. Advanced Optimization Strategies
#### Comparing Optimizers

**Task: Compare Optimizers**

Adapt your training loop into a new function, `compare_optimizers`. This function should:
1.  Take the model class as an argument.
2.  Have a dictionary of optimizers to test (e.g., `'SGD': optim.SGD`, `'Adam': optim.Adam`).
3.  Loop through this dictionary. In each iteration:
    -   Instantiate a new model and the corresponding optimizer.
    -   Run the training for a fixed number of epochs.
    -   Store the validation accuracy history for each optimizer.
4.  Return a dictionary of results.
Finally, plot the validation accuracy curves for all optimizers on a single graph to compare their convergence behavior.

#### Learning Rate Scheduling

**Task: Implement Learning Rate Scheduling**

Modify your training function to include a learning rate scheduler.

1.  After defining the optimizer, create a scheduler instance: `scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)`. This scheduler reduces the LR when the validation loss (`'min'`) stops improving for 3 (`patience`) epochs.
2.  At the end of each epoch, after the validation loop, call the scheduler's step function with the current validation loss: `scheduler.step(val_loss)`.
3.  Log the learning rate at each epoch to see how it changes over time. You can get it from `optimizer.param_groups[0]['lr']`.
4.  Plot the loss curves and the learning rate over epochs.

## Part III: Transfer Learning with Fashion-MNIST
### 1. Fashion-MNIST Introduction
Fashion-MNIST is a drop-in replacement for MNIST but is more challenging. We will use it to demonstrate the power of transfer learning.

```python
from torchvision import datasets, transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
fashion_train = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
fashion_test = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
fashion_train_loader = DataLoader(fashion_train, batch_size=64, shuffle=True)
fashion_test_loader = DataLoader(fashion_test, batch_size=64, shuffle=False)
```

### 2. Practical Implementation
We will adapt a model pre-trained on ImageNet for our grayscale clothing classification task.

**Task: Implement a Transfer Learning Model**

Your goal is to build, train, and fine-tune a transfer learning model.

1. Model Definition"

Create a `TransferLearningModel` class that inherits from `nn.Module`.

-   **`__init__(self)`**:
    -   Load a pre-trained model from `torchvision.models`, for example, `models.mobilenet_v2(pretrained=True)`.
    -   **Freeze the backbone**: Iterate through the parameters of the loaded model (`self.model.parameters()`) and set `param.requires_grad = False`. This prevents them from being updated during training.
    -   **Handle channel mismatch**: Fashion-MNIST is grayscale (1 channel), but MobileNetV2 expects RGB (3 channels). Add a `nn.Conv2d(1, 3, kernel_size=1)` layer to convert the input.
    -   **Replace the classifier**: The final layer of the pre-trained model must be replaced with a new one suited for our 10-class problem. For MobileNetV2, this is `self.model.classifier`. Replace it with a `nn.Linear` layer with the correct number of input features (1280 for MobileNetV2) and 10 output features.
-   **`forward(self, x)`**:
    -   First, pass the input `x` through your 1-to-3 channel conversion layer.
    -   Then, pass the result through the main pre-trained model.

2. Stage 1: Feature Extraction
Train the model with the frozen backbone. This means you are only training the new classifier layer you added.

- Write a training loop for this stage.
- **Important**: The optimizer should only be passed the parameters that have `requires_grad` set to true. Use `optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)`.
- Train for a few epochs (e.g., 5) with a standard learning rate like `0.001`.

3. Stage 2: Fine-Tuning
After the initial training, unfreeze some of the later layers of the pre-trained model to adapt them to the new dataset.

-   Add a method `unfreeze_layers(self, num_layers)` to your model class. It should set `requires_grad = True` for the parameters of the last `num_layers` of the feature extractor (`self.model.features`).
-   Call this method to unfreeze the last few layers (e.g., `model.unfreeze_layers(3)`).
-   Train the model again for a few more epochs.
-   **Crucially**, use a much **lower learning rate** (e.g., `1e-4`) to avoid corrupting the pre-trained weights.


### 4. Comparative Analysis
Finally, train your `SimpleCNN` from scratch on Fashion-MNIST and compare its test accuracy curve against your two-stage transfer learning model. Also, consider the difference in training time.

### Exercises for Further Practice
1.  Implement data augmentation (`torchvision.transforms`) for the CNN and measure its impact on generalization.
2.  Try different CNN architectures (e.g., add more layers, use different filter sizes).
3.  Implement early stopping in your training loop based on validation performance.
4.  Explore other pre-trained models (ResNet, EfficientNet) for transfer learning.
