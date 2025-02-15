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
