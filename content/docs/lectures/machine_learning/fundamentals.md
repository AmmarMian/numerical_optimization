---
title: 1. Machine learning fundamentals
weight: 1
---

# Machine learning fundamentals through optimization

> **Note**: This lecture bridges our study of numerical optimization with machine learning, showing how the optimization techniques we've developed provide the mathematical foundation for learning from data.

## Machine learning vs traditional programming

### Traditional programming paradigm

In traditional programming, we explicitly encode rules and logic to transform inputs into outputs. The paradigm follows a straightforward path: we receive input data, apply hand-crafted rules, and produce an output. This approach has served us well for many deterministic problems, but it encounters significant limitations when dealing with complex pattern recognition tasks.

Consider the challenge of face detection in images. A traditional approach might proceed as follows:

1. Search for oval-shaped regions (potential head detection)
2. Within these regions, identify pairs of circular areas (eye candidates)
3. Apply Hough transform for edge detection
4. Manually tune thresholds for skin color detection
5. Handle edge cases explicitly (glasses, varying lighting, different angles)

The code becomes increasingly complex as we attempt to handle more variations. Each new edge case requires additional rules, and the interactions between these rules can lead to brittle systems. More fundamentally, this approach requires the programmer to have deep domain expertise and the ability to explicitly articulate what constitutes a face in mathematical terms.

### The machine learning paradigm

Machine learning fundamentally shifts this approach. Instead of programming explicit rules, we provide examples of inputs and their corresponding outputs, and let an algorithm learn the mapping between them. The paradigm becomes:

$$\text{Input Data} + \text{Labeled Examples} \rightarrow \text{Learning Algorithm} \rightarrow \text{Parametric Model } f_{\boldsymbol{\theta}}$$

The key mathematical framework underlying this approach consists of:
- A parametric function: $f_{\boldsymbol{\theta}}: \mathcal{X} \rightarrow \mathcal{Y}$
- Parameters: $\boldsymbol{\theta} \in \mathbb{R}^p$
- Learning as optimization: finding optimal $\boldsymbol{\theta}$ to minimize a loss function

This shift brings several advantages. The system automatically learns patterns from data, naturally handles variations that appear in the training set, improves its performance with additional data, and generalizes to unseen cases that share similar patterns to the training data. The optimization perspective we've developed in previous lectures becomes the engine that drives this learning process.

## Supervised learning setup

### Mathematical formulation

In supervised learning, we begin with a training dataset:

$$\mathcal{D} = \\{(\mathbf{x}\_1, y\_1), (\mathbf{x}\_2, y\_2), \ldots, (\mathbf{x}\_n, y\_n)\\}$$

where each $\mathbf{x}\_i \in \mathbb{R}^d$ represents the input features (such as pixel values for an image or measurements for a scientific experiment), and $y\_i$ represents the corresponding output. For regression problems, $y\_i \in \mathbb{R}$, while for classification problems, $y\_i \in \\{1, 2, \ldots, K\\}$.

Our goal is to learn a function $f_{\boldsymbol{\theta}}: \mathbb{R}^d \rightarrow \mathcal{Y}$ that can accurately predict outputs for new, unseen inputs. This leads us to the core optimization problem of machine learning:

\begin{equation}
\boldsymbol{\theta}^{\*} = \arg\min_{\boldsymbol{\theta}} \frac{1}{n} \sum_{i=1}^n L(f_{\boldsymbol{\theta}}(\mathbf{x}\_i), y\_i)
\label{eq:empirical_risk}
\end{equation}

where $L(\cdot, \cdot)$ is the loss function measuring prediction error. This formulation, known as empirical risk minimization, directly connects machine learning to the optimization techniques we've studied. The choice of loss function $L$ and model family $f_{\boldsymbol{\theta}}$ determines the specific learning algorithm.

### Data splitting strategy

A fundamental principle in machine learning is that we must evaluate our model on data it hasn't seen during training. This leads to the standard practice of splitting our data into three sets:

- **Training set (70-80%)**: Used for parameter optimization via \eqref{eq:empirical_risk}
- **Validation set (10-15%)**: Used for hyperparameter tuning and model selection
- **Test set (10-15%)**: Used for final evaluation—crucially, we never touch this set during development

This splitting strategy addresses a key challenge: a model that perfectly memorizes the training data (achieving zero training loss) might perform poorly on new data. The validation set allows us to monitor this phenomenon and select models that generalize well.

## Linear and polynomial regression revisited

### Linear regression in the ML framework

Let's revisit linear regression from our optimization course, now viewing it through the machine learning lens. For the multivariate case, our model takes the form:

$$f_{\boldsymbol{\theta}}(\mathbf{x}) = \theta\_0 + \theta\_1 x\_1 + \theta\_2 x\_2 + \ldots + \theta\_d x\_d = \theta\_0 + \boldsymbol{\theta}_{1:d}^{\mathrm{T}} \mathbf{x}$$

To simplify notation, we often augment the input vector with a 1 for the bias term, allowing us to write:

$$f_{\boldsymbol{\theta}}(\mathbf{x}) = \boldsymbol{\theta}^{\mathrm{T}} \mathbf{x}$$

where now $\mathbf{x} = [1, x\_1, x\_2, \ldots, x\_d]^{\mathrm{T}}$ and $\boldsymbol{\theta} = [\theta\_0, \theta\_1, \ldots, \theta\_d]^{\mathrm{T}}$.

For the entire dataset, we can express predictions in matrix form:

$$\mathbf{f} = \mathbf{X}\boldsymbol{\theta}$$

where $\mathbf{X} \in \mathbb{R}^{n \times (d+1)}$ is the design matrix with each row being an augmented input vector.

### Mean squared error loss

The mean squared error (MSE) loss function for linear regression is:

\begin{equation}
L(\boldsymbol{\theta}) = \frac{1}{2n} \sum_{i=1}^n (y\_i - f_{\boldsymbol{\theta}}(\mathbf{x}\_i))^2 = \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2
\label{eq:mse_loss}
\end{equation}

The factor of $\frac{1}{2}$ is a convenience that simplifies the derivative. Computing the gradient:

$$\nabla_{\boldsymbol{\theta}} L = \frac{1}{n} \mathbf{X}^{\mathrm{T}}(\mathbf{X}\boldsymbol{\theta} - \mathbf{y})$$

This leads to the gradient descent update rule:

$$\boldsymbol{\theta}^{(k+1)} = \boldsymbol{\theta}^{(k)} - \alpha \nabla_{\boldsymbol{\theta}} L = \boldsymbol{\theta}^{(k)} - \frac{\alpha}{n} \mathbf{X}^{\mathrm{T}}(\mathbf{X}\boldsymbol{\theta}^{(k)} - \mathbf{y})$$

Notice how this is exactly the optimization problem from our previous lectures, now applied to learning from data. The line search methods we studied for choosing $\alpha$ apply directly here.

### Polynomial regression and model complexity

We can extend linear regression to capture nonlinear relationships by using polynomial features. For a single input variable $x$, we might use:

$$f_{\boldsymbol{\theta}}(x) = \theta\_0 + \theta\_1 x + \theta\_2 x^2 + \ldots + \theta\_p x^p$$

More generally, we can use any set of basis functions $\phi\_j(x)$:

$$f_{\boldsymbol{\theta}}(\mathbf{x}) = \sum_{j=0}^p \theta\_j \phi\_j(\mathbf{x})$$

This remains linear in the parameters $\boldsymbol{\theta}$, so we can still use our linear regression machinery. However, increasing model complexity introduces a fundamental trade-off:

{{<definition "Bias-variance trade-off" bias_variance>}}
**Underfitting** occurs when the model is too simple to capture the underlying patterns (high bias, low variance). **Overfitting** occurs when the model memorizes the training data, including noise, leading to poor generalization (low bias, high variance). The goal is to find the optimal complexity that minimizes the total error on new data.
{{</definition>}}

As we increase polynomial degree:
- **Degree 1 (linear)**: May underfit if the true relationship is nonlinear
- **Degree 2-4**: Often captures real patterns well
- **High degree (>10)**: Risk of overfitting—the model fits training data perfectly but generalizes poorly

The validation set becomes crucial here: we monitor validation error as we increase model complexity and select the degree that minimizes validation error, not training error.

<figure>
  <img src="https://upload.wikimedia.org/wikipedia/commons/9/9f/Bias_and_variance_contributing_to_total_error.svg"  alt="Bias-variance trade-off" style="width: 600px; margin: auto;">
  <figcaption>Bias-variance trade-off</figcaption>
</figure>


## Logistic regression for classification

Now that we've established the foundations of linear regression, we turn to classification problems, where the output $y$ is categorical rather than continuous. The most common approach for binary classification is logistic regression.

### Why not linear regression for classification?

For binary classification where $y \in \\{0, 1\\}$, we might initially consider using linear regression. However, this approach has several problems:

1. Linear regression can output any real number, not just values in $[0,1]$
2. The loss function doesn't match the problem—squared error penalizes "very correct" predictions
3. The model is sensitive to outliers in a way that doesn't make sense for classification

We need a model that outputs probabilities and a loss function suited to classification.

### The sigmoid function

The sigmoid (logistic) function transforms any real number to the interval $(0,1)$:

\begin{equation}
\sigma(z) = \frac{1}{1 + e^{-z}}
\label{eq:sigmoid}
\end{equation}


### The logistic regression model

Logistic regression models the probability of the positive class as:

\begin{equation}
P(y=1|\mathbf{x}) = \sigma(\boldsymbol{\theta}^{\mathrm{T}} \mathbf{x}) = \frac{1}{1 + e^{-\boldsymbol{\theta}^{\mathrm{T}} \mathbf{x}}}
\label{eq:logistic_model}
\end{equation}

Correspondingly:
$$P(y=0|\mathbf{x}) = 1 - P(y=1|\mathbf{x}) = \frac{e^{-\boldsymbol{\theta}^{\mathrm{T}} \mathbf{x}}}{1 + e^{-\boldsymbol{\theta}^{\mathrm{T}} \mathbf{x}}}$$

For prediction, we typically use the decision rule:
$$\hat{y} = \begin{cases}
1 & \text{if } P(y=1|\mathbf{x}) > 0.5 \\\\
0 & \text{otherwise}
\end{cases}$$

This corresponds to the decision boundary $\boldsymbol{\theta}^{\mathrm{T}} \mathbf{x} = 0$. Despite using a nonlinear sigmoid function, the decision boundary remains linear in the input space—the sigmoid simply maps the linear function to probabilities.

### Maximum likelihood and cross-entropy loss

To derive the appropriate loss function, we use the principle of maximum likelihood. For a single example, the likelihood is:

$$P(y|\mathbf{x}) = \sigma(\boldsymbol{\theta}^{\mathrm{T}} \mathbf{x})^y \cdot (1-\sigma(\boldsymbol{\theta}^{\mathrm{T}} \mathbf{x}))^{1-y}$$

This compact expression gives $P(y=1|\mathbf{x})$ when $y=1$ and $P(y=0|\mathbf{x})$ when $y=0$.

Taking the negative log-likelihood over all training examples gives us the cross-entropy loss:

\begin{equation}
L(\boldsymbol{\theta}) = -\frac{1}{n} \sum_{i=1}^n \left[y\_i \log(\hat{y}\_i) + (1-y\_i)\log(1-\hat{y}\_i)\right]
\label{eq:cross_entropy}
\end{equation}

where $\hat{y}\_i = \sigma(\boldsymbol{\theta}^{\mathrm{T}} \mathbf{x}\_i)$.

### Gradient computation for logistic regression

Computing the gradient of the cross-entropy loss yields a remarkably clean result. For the $j$-th component:

$$\frac{\partial L}{\partial \theta\_j} = \frac{1}{n} \sum_{i=1}^n (\hat{y}\_i - y\_i) x\_{i,j}$$

In vector form:
\begin{equation}
\nabla_{\boldsymbol{\theta}} L = \frac{1}{n} \mathbf{X}^{\mathrm{T}} (\hat{\mathbf{y}} - \mathbf{y})
\label{eq:logistic_gradient}
\end{equation}

This has exactly the same form as the gradient for linear regression! The only difference is that $\hat{\mathbf{y}}$ contains sigmoid-transformed predictions rather than linear predictions. This elegant connection shows how different loss functions and model choices lead to similar optimization structures.

### Connection to optimization theory

Logistic regression loss is convex in $\boldsymbol{\theta}$, ensuring that any local minimum is a global minimum. The Hessian is:

$$\mathbf{H} = \frac{1}{n} \mathbf{X}^{\mathrm{T}} \mathbf{S} \mathbf{X}$$

where $\mathbf{S}$ is a diagonal matrix with $S\_{ii} = \hat{y}\_i(1-\hat{y}\_i)$. Since $\hat{y}\_i \in (0,1)$, all diagonal elements are positive, making $\mathbf{H}$ positive semi-definite. This connects directly to the second-order sufficient conditions for optimality from our previous lectures.

## Summary and next steps

We've seen how machine learning problems naturally formulate as optimization problems. Linear regression minimizes squared error, while logistic regression maximizes likelihood (minimizes cross-entropy). Both lead to optimization problems solvable with the gradient-based methods from our previous lectures.

In the next lecture, we'll explore how to scale these methods to massive datasets through stochastic gradient descent and its variants, connecting to the convergence theory and step size selection strategies we've already developed.
