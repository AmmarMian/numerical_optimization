---
title: I - Linear Regression models
weight: 10
---

# Linear Regression models

## Introduction

In this lab session, we will explore the fundamental concepts of numerical optimization through the lens of linear regression. We'll begin with the simplest case and gradually build up to more complex scenarios, comparing analytical solutions with numerical methods at each step.

Linear regression is perhaps the most fundamental problem in machine learning and statistics. While it has a closed-form solution, implementing numerical optimization methods for this problem provides excellent intuition for more complex optimization scenarios where analytical solutions don't exist.

## Learning objectives

By the end of this session, you should be able to:
- Derive the analytical solution for simple linear regression
- Implement gradient descent with various step size strategies
- Understand the connection between the one-dimensional and multi-dimensional cases
- Apply line search techniques to improve convergence


## I - One dimensional case

Let us first start with the form that most people are familiar with, the linear regression model in one dimension. The setup is as follows:
* We have a set of data points $\\{(x_i, y_i)\\}_{i=1}^n$. Here the $x_i$ are the input features and the $y_i$ are the target values.
* Assuming there is a linear relationship between and target because of some underlying phenomenon, we model the observations as:
\begin{equation}
y_i = \alpha x_i + \beta + \epsilon
\label{eq:linear_model_1d}
\end{equation}
where $\\epsilon$ is a random noise term that we assume to be normally distributed with mean 0 and variance $\\sigma^2$.


Our goal is then to find the parameters $\\alpha$ and $\\beta$ that "best match" the data points. Such a program can is illustrated with following [interactive plot]("../../../interactive/linear_regression_1d.html").

### 1. Modeling and solving the problem

1. Propose a loss function that quantifies the difference between the observed $y_i$ and the predicted values $\\hat{y}_i = \\alpha x_i + \\beta$.

{{<hiddenhint "Hint">}}
The most common loss function for regression problems is the mean squared error (MSE):
$$
L(\\alpha, \\beta) = \\frac{1}{2} \\sum_{i=1}^n (y_i - (\\alpha x_i + \\beta))^2
$$
{{</hiddenhint>}}

2. Show that the loss function is convex in the parameters $\\alpha$ and $\\beta$.

{{<hiddenhint "Hint">}}
To show convexity, we need to demonstrate that the Hessian matrix of second derivatives is positive semi-definite. Or that the function is a positive linear combination of convex functions.
The loss function is a quadratic function in $\\alpha$ and $\\beta$, which is convex. The Hessian matrix will have positive eigenvalues, confirming convexity.
{{</hiddenhint>}}

3. Derive the analytical solution for the parameters $\\alpha$ and $\\beta$ by setting the gradients of the loss function with respect to these parameters to zero.

{{<hiddenhint "Hint">}}
It is often useful to express the gradients in terms of the means and variances of the data points:
* means  : $$\\bar{x} = \\frac{1}{n} \\sum_{i=1}^n x_i, \\quad \\bar{y} = \\frac{1}{n} \\sum_{i=1}^n y_i$$
* variance: $$s_{xx} = \\frac{1}{n} \\sum_{i=1}^n (x_i - \\bar{x})^2$$
* covariance: $$s_{xy} = \\frac{1}{n} \\sum_{i=1}^n (x_i - \\bar{x})(y_i - \\bar{y})$$
{{</hiddenhint>}}

4. Implement the analytical solution in Python and compute the optimal parameters for a given dataset. To generate dataset, you can use following code snippet:

```python
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
rng = np.random.default_rng(42)

# Generate synthetic data
n_samples = 50
x = np.linspace(0, 10, n_samples)
# True parameters
alpha = 2.5
beta = 1.0
# Add Gaussian noise
noise = rng.normal(0, 1, n_samples)
y = alpha * x + beta + noise

# Visualize the data
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.7, label='Data points')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Synthetic linear data with noise')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
```

5. (Bonus) Show that doing a Maximum Likelihood Estimation (MLE) for the parameters $\\alpha$ and $\\beta$ leads to the same solution as minimizing the loss function derived above.

{{<hiddenhint "Hint">}}
The MLE for the parameters in a linear regression model with Gaussian noise leads to minimizing the negative log-likelihood, which is equivalent to minimizing the mean squared error loss function. The derivation involves taking the logarithm of the Gaussian probability density function and simplifying it, leading to the same equations for $\\alpha$ and $\\beta$ as derived from the loss function.
{{</hiddenhint>}}

### 2. Gradient descent for the one-dimensional case

Now that we have a good understanding of the problem and have implemented the analytical solution, let's explore how we can solve this problem using numerical optimization techniques, specifically steepest gradient descent, i.e always taking the gradient as the direction.

1. Recalling the update rule for steepest gradient descent of a point $\\theta_k$ at iteration $k$:
$$
\\theta_{k+1} = \\theta_k - \\alpha_k \\nabla L(\\theta_k)
$$
where $\\alpha_k$ is the step size at iteration $k$,

give the update rule for the parameters $\\alpha$ and $\\beta$ in the context of our linear regression problem.

{{<hiddenhint "Hint">}}
The update rules for the parameters $\\alpha$ and $\\beta$ can be derived from the gradients of the loss function:
$$
\\begin{align*}
\\alpha_{k+1} &= \\alpha_k - \\alpha_k \\frac{\\partial L}{\\partial \\alpha}(\\alpha_k, \\beta_k) \\\\
\\beta_{k+1} &= \\beta_k - \\alpha_k \\frac{\\partial L}{\\partial \\beta}(\\alpha_k, \\beta_k)
\\end{align*}
$$
where the gradients are computed as follows:
$$
\\begin{align*}
\\frac{\\partial L}{\\partial \\alpha} &= -\\sum_{i=1}^n (y_i - (\\alpha_k x_i + \\beta_k)) x_i \\\\
\\frac{\\partial L}{\\partial \\beta} &= -\\sum_{i=1}^n (y_i - (\\alpha_k x_i + \\beta_k))
\\end{align*}
$$
{{</hiddenhint>}}

2. Implement gradient descent with a constant step size $\\alpha_k = \\alpha$ for all iterations. Your function should:
   - Take initial parameters $(\\alpha_0, \\beta_0)$, step size $\\alpha$, and number of iterations as inputs.
   - Return the trajectory of parameters and loss values.
   - Include a stopping criterion based on gradient magnitude.


3. Experiment with different step sizes: $\\alpha \\in \\{0.0001, 0.001, 0.01, 0.1\\}$. Plot the loss function over iterations for each case. What do you observe?

{{<hiddenhint "Hint">}}
The loss function should decrease over iterations, but the rate of decrease will depend on the step size. A very small step size will lead to slow convergence, while a very large step size may cause divergence or oscillations.
{{</hiddenhint>}}

4. For a fixed number of iterations (say 1000), plot the final error as a function of step size on a logarithmic scale. What is the optimal range for $\\alpha$?

{{<hiddenhint "Hint">}}
The optimal range for $\\alpha$ is typically small enough to ensure convergence but large enough to allow for reasonable speed of convergence. You may find that values around $0.001$ to $0.01$ work well, but this can depend on the specific dataset and problem.
{{</hiddenhint>}}

5. Let's try a first experiment with a decreasing step size. Implement a linear decay strategy:
$$
\\alpha_k =  \\alpha_0 - k \\cdot \\gamma,
$$
where $\\gamma$ is a small constant (e.g., $0.0001$). Experiment with different values of $\\alpha_0$ and $\\gamma$. 

Compare the convergence behavior with constant step size. Plot the loss function and parameter trajectories over iterations.

6. Why might decreasing step sizes be beneficial? What are the trade-offs between aggressive and conservative decay rates?

{{<hiddenhint "Hint">}}
Decreasing step sizes can help avoid overshooting the minimum and allow for finer adjustments as the algorithm converges.

In nonconvex problmes, aggressive decay rates may lead to faster convergence initially but can cause the algorithm to get stuck in local minima, while conservative rates may lead to slower convergence but better exploration of the parameter space.
{{</hiddenhint>}}

7. Try implementing an exponential decay strategy:
$$
\\alpha_k = \\alpha_0 \\cdot \\gamma^k$$
where $\\gamma \\in (0, 1)$ is the decay rate. Experiment with different values of $\\gamma$ (e.g., $0.9$, $0.95$, $0.99$) and compare the convergence behavior with constant and linear decay strategies.

{{<hiddenhint "Hint">}}
Exponential decay is more aggressive, thus it may also cause the step size to become too small too quickly, leading to slow convergence in later iterations. The choice of $\\gamma$ can significantly affect the convergence behavior.
{{</hiddenhint>}}


## II - Multiple variables case

Now that we have a good understanding of the one-dimensional case, let's generalize our approach to multiple dimensions. The setup is similar, but now we have multiple features and parameters:
* We have a set of data points
$\\{(\mathbf{x}_i, y_i)\\}\_{i=1}^n$, where $\\mathbf{x}_i \\in \\mathbb{R}^d$ are the input features and $y_i \\in \\mathbb{R}$ are the target values.
* We model the observations as:
\begin{equation}
y_i = \\mathbf{w}^{\\mathrm{T}} \\mathbf{x}_i + \beta + \\epsilon
\label{eq:linear_model_d}
\end{equation}
where $\\mathbf{w} \\in \\mathbb{R}^d$ is the weight vector, $\beta \\in \\mathbb{R}$ is the bias term, and $\\epsilon$ is a random noise term that we assume to be normally distributed with mean 0 and variance $\\sigma^2$.

{{<center>}}
{{<NumberedFigure
  src=" ../../../../../tikZ/2D_linear/main.svg"
  alt="2D linear regression"
  width="400px"
  caption="2D linear regression"
  label="linear_regression_2d"
>}}
{{</center>}}

An example of this model is illustrated in Figure {{<NumberedFigureRef linear_regression_2d>}}, where the data points are represented in a two-dimensional space, and the linear regression model fits a plane to the data.


This model is more general than the one-dimensional case, as it allows for multiple features to influence the target variable. Notably, we can also augment the feature vectors with a constant term to simplify the notation so that we don't have to deal with the bias term separately. We define:
\begin{equation}
\\tilde{\\mathbf{x}}_i = [1, \\mathbf{x}_i^{\\mathrm{T}}]^{\\mathrm{T}} \\in \\mathbb{R}^{d+1}
\end{equation}
and
\begin{equation}
\\tilde{\\mathbf{w}} = [\beta, \\mathbf{w}^{\\mathrm{T}}]^{\\mathrm{T}} \\in \\mathbb{R}^{d+1}
\end{equation}
so that we can rewrite the model as:
\begin{equation}
y_i = \\tilde{\\mathbf{w}}^{\\mathrm{T}} \\tilde{\\mathbf{x}}_i + \\epsilon
\label{eq:linear_model_d_augmented}
\end{equation}

Our goal is then to find the parameters $\\mathbf{w}$ and $\beta$ that "best match" the data points, or in augmented notation, to find $\\tilde{\\mathbf{w}}$ that minimizes the loss function.

### 1. Modeling and solving the problem

While the augmented formulation is nice, we can also express the model in matrix form for the observed data. We define the design matrix $\\mathbf{X} \\in \\mathbb{R}^{n \\times (d+1)}$ as:
\begin{equation}
\\mathbf{X} = \\begin{bmatrix}
1 & x_{11} & x_{12} & \\ldots & x_{1d} \\\\
1 & x_{21} & x_{22} & \\ldots & x_{2d} \\\\
\\vdots & \\vdots & \\vdots & \\ddots & \\vdots \\\\
1 & x_{n1} & x_{n2} & \\ldots & x_{nd}
\\end{bmatrix}
\end{equation}
and the target vector $\\mathbf{y} \\in \\mathbb{R}^n$ as:
\begin{equation}
\\mathbf{y} = \\begin{bmatrix}
y_1 \\\\
y_2 \\\\
\\vdots \\\\
y_n
\\end{bmatrix}
\end{equation}

Then we can express the model as:
\begin{equation}
\\mathbf{y} = \\mathbf{X} \\tilde{\\mathbf{w}} + \\boldsymbol{\\epsilon},
\label{eq:linear_model_matrix}
\end{equation}

where $\\boldsymbol{\\epsilon} \\in \\mathbb{R}^n$ is the noise vector. This more compact formulation is interesting for several reasons:
* It already encapsulates the observed data in the model and we consider all the  $y_i$ as a vector, which allows us to work with the entire dataset at once.
* the matrix form allow us to obtain solutions that will be expressed as matrix operations, which is more efficient for larger datasets
* it allows us to use linear algebra techniques to derive the solution

1. Propose a loss function that quantifies the difference between the observed $y_i$ and the predicted values $\\hat{y}_i = \\mathbf{X} \\tilde{\\mathbf{w}}$.

{{<hiddenhint "Hint">}}
The most common loss function for regression problems is the mean squared error (MSE):
$$
L(\\tilde{\\mathbf{w}}) = \\frac{1}{2} \\|\\mathbf{y} - \\mathbf{X} \\tilde{\\mathbf{w}}\\|^2_2
$$
{{</hiddenhint>}}

2. Show that the loss function is convex in the parameters $\\tilde{\\mathbf{w}}$.

{{<hiddenhint "Hint">}}
As in the one-dimensional case, the loss function is a quadratic function in $\\tilde{\\mathbf{w}}$, which is convex. The Hessian matrix of second derivatives will be positive semi-definite, confirming convexity.
{{</hiddenhint>}}

3. Derive the gradient of the loss function with respect to $\\tilde{\\mathbf{w}}$ in matrix form. To help yourselves, you can use the properties of matrix derivatives from matrix cookbook available [here](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) and identity of vector norms:
$$
\lVert \mathbf{u} - \mathbf{v} \rVert^2_2 = \mathbf{u}^{\mathrm{T}} \mathbf{u} - 2 \mathbf{u}^{\mathrm{T}} \mathbf{v} + \mathbf{v}^{\mathrm{T}} \mathbf{v}.
$$

and show that optimal solution $\\tilde{\\mathbf{w}}$ satisfies the normal equations:
$$
\\mathbf{X}^{\\mathrm{T}} \\mathbf{X} \\tilde{\\mathbf{w}} = \\mathbf{X}^{\\mathrm{T}} \\mathbf{y}.
$$

{{<hiddenhint "Hint">}}
Make use of 
* $\\frac{\\partial}{\\partial \\mathbf{x}} \\mathbf{x}^{\\mathrm{T}} \\mathbf{a} = \\frac{\\partial}{\\partial \\mathbf{x}} \\mathbf{a}^{\\mathrm{T}} \\mathbf{x} = \\mathbf{a}$
* $\\frac{\\partial}{\\partial \\mathbf{x}} \\|\\mathbf{x}\\|^2_2 = \\frac{\\partial}{\\partial \\mathbf{x}} \\mathbf{x}^{\\mathrm{T}} \\mathbf{x} = 2 \\mathbf{x}$
* $\\frac{\\partial}{\\partial \\mathbf{x}} \\lVert \\mathbf{A} \\mathbf{x} \\rVert^2_2 = 2 \\mathbf{A}^{\\mathrm{T}} \\mathbf{A} \\mathbf{x}$
{{</hiddenhint>}}

Thus, to obtain optimal parameters $\\tilde{\\mathbf{w}}$, we can solve the normal equations:
$$
\\tilde{\\mathbf{w}} = (\\mathbf{X}^{\\mathrm{T}} \\mathbf{X})^{-1} \\mathbf{X}^{\\mathrm{T}} \\mathbf{y}.
$$

> Note: The matrix $\\mathbf{X}^{\\mathrm{T}} \\mathbf{X}$ is known as the Gram matrix, and it is positive semi-definite. If $\\mathbf{X}$ has full column rank, then $\\mathbf{X}^{\\mathrm{T}} \\mathbf{X}$ is invertible, and we can compute the unique solution for $\\tilde{\\mathbf{w}}$. Otherwise, the solution is not unique, and we may need to use regularization techniques (e.g., ridge regression) to obtain a stable solution, but we will not cover this in this lab.

4. Implement the analytical solution using NumPy's linear algebra functions. Compare your result with `np.linalg.lstsq`. To generate dataset, you can use following code snippet:

```python
import numpy as np

# Generate multi-dimensional data
d = 5  # number of features
n_samples = 100

# Generate random features
X = np.random.randn(n_samples, d)
# Add intercept term
X_augmented = np.column_stack([np.ones(n_samples), X])
# True parameters
w_true = np.random.randn(d + 1)  # including bias
# Generate targets with noise
y = X_augmented @ w_true + 0.5 * np.random.randn(n_samples)
print(f"Data shape: {X.shape}")
print(f"Augmented data shape: {X_augmented.shape}")

print(f"True parameters: {w_true}")
```


### 2. Gradient descent for the multiple variables case

Rather than inverting the matrix $\\mathbf{X}^{\\mathrm{T}} \\mathbf{X}$, which can be computationally expensive for large datasets, we can use gradient descent to find the optimal parameters $\\tilde{\\mathbf{w}}$.

1. Give the update rule for the parameters $\\tilde{\\mathbf{w}}$ in the context of our linear regression problem using steepest gradient descent.

{{<hiddenhint "Hint">}}
The update rule for the parameters $\\tilde{\\mathbf{w}}$ can be derived from the gradient of the loss function:
$$
\\tilde{\\mathbf{w}}_{k+1} = \\tilde{\\mathbf{w}}_k - \\alpha_k \\nabla L(\\tilde{\\mathbf{w}}_k)
$$
where the gradient is given by:
$$
\\nabla L(\\tilde{\\mathbf{w}}) = -\\mathbf{X}^{\\mathrm{T}} (\\mathbf{y} - \\mathbf{X} \\tilde{\\mathbf{w}})
$$
{{</hiddenhint>}}

2. Implement gradient descent with a constant step size $\\alpha_k = \\alpha$ for all iterations. Your function should:
   - Take initial parameters $\\tilde{\\mathbf{w}}_0$, step size $\\alpha$, and number of iterations as inputs.
   - Return the trajectory of parameters and loss values.
   - Include a stopping criterion based on gradient magnitude.

### 3. Experimenting with  backtracking line search

From implementing gradient descent, we have seen that the choice of step size $\\alpha$ can significantly affect the convergence behavior. A fixed step size may not be optimal for all iterations, leading to slow convergence or oscillations. On the other hand, a decreasing step size is not the best choice as it may lead to very small step sizes in later iterations, causing slow convergence. Let us put in practice the theory we have set in place around line search techniques to adaptively choose the step size at each iteration.

1. Implement a backtracking line search algorithm to adaptively choose the step size $\\alpha_k$ at each iteration. For a reminder, check the memo [here]({{<ref "backtracking.md">}}).

2. Use the backtracking line search to find the optimal step size for each iteration of gradient descent. Compare the convergence behavior with constant and decreasing step sizes.

3. Experiment with different parameters for the backtracking line search, such as the initial step size $\\alpha_0$, reduction factor $\\rho$, and Armijo parameter $c_1$. How do these parameters affect the convergence behavior?

{{<hiddenhint "Hint">}}
The backtracking line search will adaptively adjust the step size based on the Armijo condition, allowing for more efficient convergence. The choice of $\\alpha_0$, $\\rho$, and $c_1$ can significantly affect the speed of convergence and stability of the algorithm.
{{</hiddenhint>}}

### 4. Using more complex linesearch techniques using toolboxes

In practice, we often use more sophisticated line search techniques that are not so easy to implement from scratch. One such technique is the `line_search` function from SciPy's optimization module, which implements  interpolation techniques to find an optimal step size.

1. Use the `line_search` function from SciPy's optimization module to find the optimal step size for each iteration of gradient descent. Compare the convergence behavior with the backtracking line search.
Documentation is available [here](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.line_search.html).

## III - (Bonus) The general case

Consider the general case where the target is also a vector, i.e., we have a multi-output linear regression problem. The model can be expressed as:
\begin{equation}
\\mathbf{Y} = \\mathbf{X} \\tilde{\\mathbf{W}} + \\boldsymbol{\\epsilon},
\label{eq:linear_model_multi_output}
\end{equation}
where $\\mathbf{Y} \\in \\mathbb{R}^{n \\times m}$ is the target matrix with $m$ outputs, and $\\tilde{\\mathbf{W}} \\in \\mathbb{R}^{(d+1) \\times m}$ is the weight matrix.

Derive all the necessary tools to solve this problem using the same techniques as in the previous sections.

## IV - (Bonus) Regularization

In practice, we often encounter situations where the model is overfitting the data, especially in high-dimensional settings. Regularization techniques are used to prevent overfitting by adding a penalty term to the loss function.

Implement L2 regularization (ridge regression) by adding a penalty term to the loss function:
\begin{equation}
L(\\tilde{\\mathbf{w}}) = \\frac{1}{2} \\|\\mathbf{y} - \\mathbf{X} \\tilde{\\mathbf{w}}\\|^2_2 + \\frac{\\lambda}{2} \\|\\tilde{\\mathbf{w}}\\|^2_2,
\end{equation}
where $\\lambda > 0$ is the regularization parameter.

## V - (Bonus) Nonlinear regression

It's actually possible to extend the linear regression model to nonlinear regression by setting up the design matrix $\\mathbf{X}$ to include nonlinear features of the input data. For example, we can include polynomial features suc as:
\begin{equation}
\\mathbf{X} = \\begin{bmatrix}
1 & x_1 & x_1^2 & \\ldots & x_1^d \\\\
1 & x_2 & x_2^2 & \\ldots & x_2^d \\\\
\\vdots & \\vdots & \\vdots & \\ddots & \\vdots \\\\
1 & x_n & x_n^2 & \\ldots & x_n^d
\\end{bmatrix}
\end{equation}

From this information and your own research , implement a nonlinear regression model using polynomial features. You can use the `PolynomialFeatures` class from `sklearn.preprocessing` to generate polynomial features.
