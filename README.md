# Python Labs: Hands-On Computational Experiments

This repository is a curated set of interactive Python labs, designed to explore fundamental and applied topics in computational science. Each lab introduces a new concept through practical coding exercises, using popular Python libraries such as NumPy, Matplotlib, and more.

> This project is a work in progress — additional labs in Data Science, ML, and Scientific Computing will be added regularly.

---

## Lab 1: Foundations of Linear Algebra with NumPy

In this first lab, we dive into the core building blocks of Linear Algebra using NumPy. You'll work with matrices and vectors, perform algebraic transformations, and apply powerful operations like Singular Value Decomposition (SVD) to tasks like image compression.

- Perform operations on vectors and matrices (dot, transpose, inverse)
- Use NumPy's linalg tools for decomposition and system solving
- Apply SVD to compress and reconstruct images
- Understand eigenvalues and eigenvectors with visual interpretation

---

## Operations Covered

| Category             | Function/Usage                     |
|----------------------|------------------------------------|
| Transpose            | A.T                                |
| Dot Product          | np.dot(A, B) or A @ B              |
| Outer Product        | np.outer(a, b)                     |
| Inverse              | np.linalg.inv(A)                   |
| Determinant          | np.linalg.det(A)                   |
| Power of Matrix      | np.linalg.matrix_power(A, n)       |
| Eigenvalues/Vectors  | np.linalg.eig(A)                   |
| Frobenius Norm       | np.linalg.norm(A)                  |
| Row-wise Norm        | np.linalg.norm(A, axis=1)          |
| Matrix Trace         | np.trace(A)                        |
| Linear Solver        | np.linalg.solve(A, B)              |

---

## Project: Image Compression with SVD

We use SVD to reduce an image’s dimensionality while retaining its visual quality.

```python
from skimage import data
import numpy as np
import matplotlib.pyplot as plt

# Load sample image and normalize
img = data.cat() / 231.0
img = np.transpose(img, (2, 0, 1))

# SVD decomposition
U, S, V = np.linalg.svd(img)
Sigma = np.zeros((3,300,451)
for i in range(3):
    np.fill_diagonal(Sigma[i], S[i])

# Reconstruction with reduced components
k = 25
reconstructed = U @ Sigma[:, :, :k] @ V[:, :k, :]
reconstructed = np.transpose(reconstructed, (1, 2, 0))

plt.imshow(reconstructed)
plt.axis('off')
plt.show()
```

## Lab2:- 
Statistics and Probability: Fundamentals
---
### Random Events & Simulations
- Simulating coin flips and dice rolls
- Visualizing outcomes with pie charts and bar plots

### Probability Distributions
- Working with **Bernoulli**, **Binomial**, and **Poisson** distributions
- Plotting **Normal distributions**
- Demonstrating the **Central Limit Theorem**

### Theory vs Experiment
- Comparing empirical results with theoretical expectations
- Concept of **Regression to the Mean**

### Card-Based Probability
- Drawing cards with/without replacement
- Exploring cut-deck probabilities

### Expected Values
- Simulating repeated trials to understand expected outcomes
- Examples: drawing cards until an Ace appears

### Measurement Errors
- Comparing squared and absolute errors
- Visual representation of Mean Squared Error

### Z-Scores & Standardization
- Normalizing scores from multiple tests (e.g. Math, Bio, Physics)
- Creating comparable distributions

### Random Variables
- Discrete vs Continuous variables
- Working with `scipy.stats` for CDFs, PDFs, and quantiles
