# Logistic Regression from Scratch

This project implements a simple Logistic Regression model from scratch in Python. The goal is to understand how logistic regression works under the hood without using high-level libraries like scikit-learn for the algorithm itself.

## Overview

Logistic Regression is used for binary classification. It models the probability that a given input belongs to a specific class using the logistic (sigmoid) function. The model is defined by the equation:

$$
z = w \cdot x + b
$$

The sigmoid function is then applied to $z$ to produce a probability:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

The output $\sigma(z)$ is interpreted as the probability that the input belongs to the positive class (1). A threshold of 0.5 is typically used to decide the class label.

## Code Structure

This folder contains two main files:

- **regression.py** – Contains the implementation of the Logistic Regression model.
- **test.py** – Demonstrates how to use the model with the Breast Cancer dataset from scikit-learn.

---

## Math

Instead of Mean Squared Error, we use a cost function called Cross-Entropy, also known as Log Loss. Cross-entropy loss can be divided into two separate cost functions: one for y=1 and one for y=0.
![image](https://github.com/user-attachments/assets/4ae60e1c-efbd-4a12-9731-5506762010c0)

The benefits of taking the logarithm reveal themselves when you look at the cost function graphs for y=1 and y=0. These smooth monotonic functions (always increasing or always decreasing) make it easy to calculate the gradient and minimize cost. Image from Andrew Ng’s slides on logistic regression.
![image](https://github.com/user-attachments/assets/5c11c5e6-208c-481e-ad2c-8d8c9497da02)

The key thing to note is the cost function penalizes confident and wrong predictions more than it rewards confident and right predictions! The corollary is increasing prediction accuracy (closer to 0 or 1) has diminishing returns on reducing cost due to the logistic nature of our cost function.



## regression.py

```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        # Convert probabilities to class labels (0 or 1)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
```
