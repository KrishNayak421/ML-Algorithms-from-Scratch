# Linear Regression from Scratch

## Overview

Linear regression is a fundamental machine learning algorithm used to model the relationship between a dependent variable (target) and an independent variable (feature) by fitting a linear equation to the data. The goal is to find the best-fitting straight line, defined by the equation:

$$
\hat{y} = w \times x + b
$$

where:

- **\( w \)** is the slope (weight),
- **\( b \)** is the intercept (bias), and
- **\( \hat{y} \)** is the predicted output.

The quality of the fit is usually measured using the **Mean Squared Error (MSE)**, which is calculated as:

$$
\text{MSE} = \frac{1}{n} \sum\_{i=1}^{n} \left(y_i - (w \times x_i + b)\right)^2
$$

## How It Works

To minimize the error (MSE), we use **gradient descent**, an iterative optimization algorithm that updates the parameters \( w \) and \( b \) in the direction of the steepest descent (i.e., the negative gradient). The update rules are:

$$
w := w - \alpha \frac{\partial \text{MSE}}{\partial w}, \quad b := b - \alpha \frac{\partial \text{MSE}}{\partial b}
$$

where $\( \alpha \)$ is the learning rate.

## My Implementation

The code below demonstrates a simple linear regression model built from scratch using Python. It:

- Reads a dataset from a CSV file.
- Defines a loss function to compute the Mean Squared Error.
- Implements gradient descent to update model parameters.
- Trains the model for a specified number of epochs.
- Visualizes the data and the regression line.

### Code Explanation

```python
import matplotlib.pyplot as plt
import pandas as pd
import time

# Record start time
start = time.time()

# Load training data from CSV
data = pd.read_csv("train.csv")

# Loss function: calculates the Mean Squared Error (MSE)
def loss_function(w, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].feature    # Access the feature value
        y = points.iloc[i].target     # Access the target value
        total_error += (y - (w * x + b))**2
    total_error = total_error / float(len(points))
    return total_error

# Gradient descent function: computes gradients and updates parameters
def gradient_descent(w_now, b_now, points, L):
    w_gradient = 0
    b_gradient = 0
    n = len(points)
    for i in range(n):
        x = points.iloc[i].feature
        y = points.iloc[i].target
        w_gradient += (-2 / n) * x * (y - (w_now * x + b_now))
        b_gradient += (-2 / n) * (y - (w_now * x + b_now))

    # Update parameters using the learning rate L
    w = w_now - w_gradient * L
    b = b_now - b_gradient * L

    return w, b

# Initialize parameters, learning rate, and number of epochs
w = 0
b = 0
L = 0.0001
epochs = 1000

# Training loop: update w and b using gradient descent
for i in range(epochs):
    w, b = gradient_descent(w, b, data, L)
    if i % 50 == 0:
        print("Epoch:", i, "Loss:", loss_function(w, b, data))
print("Final parameters:", w, b)

# Visualize the results: scatter plot for data and line for the regression fit
plt.scatter(data.feature, data.target, color="red", label="Data")
plt.plot(range(0, 100), [w * x + b for x in range(0, 100)], color="black", label="Fit")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Linear Regression Fit")
plt.legend()
plt.show()

# Print the total time taken
end = time.time()
print("Time taken:", end - start)
```
