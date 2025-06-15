---
title: "2. Classification and support vector machines"
weight: 2
chapter: 2
---

# Classification and support vector machines

> **Note**: This lecture extends our optimization framework to classification problems, introducing the perceptron algorithm and support vector machines (SVMs). We'll see how the constrained optimization techniques from previous lectures lead to powerful classification methods.

## Classification problems in depth

### Types of classification

In our previous lecture, we introduced binary classification through logistic regression. Let's now formalize the broader classification framework and understand how different formulations connect to optimization problems.

**Binary classification** remains the fundamental building block. We can choose between two label conventions:
- $y \in \\{0, 1\\}$: Natural for probabilistic models like logistic regression
- $y \in \\{-1, +1\\}$: Convenient for geometric algorithms like perceptron and SVM

The choice affects our loss functions and update rules, but the underlying problem remains the same: partition the input space into two regions.

**Multiclass classification** extends to $K > 2$ classes with labels $y \in \\{1, 2, \ldots, K\\}$. We often use one-hot encoding, transforming each label into a vector $\mathbf{y} \in \\{0,1\\}^K$ where exactly one component equals 1. For instance, in digit recognition with $K=10$, the digit "3" becomes $\mathbf{y} = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]^{\mathrm{T}}$. This encoding naturally extends to probabilistic models where we predict a distribution over classes.

### Performance metrics for classification

Classification performance requires more nuanced evaluation than regression's simple squared error. Consider a binary classifier's predictions compared to true labels, summarized in the confusion matrix:

```
              Predicted
              0    1
Actual   0   TN   FP
         1   FN   TP
```

where TN = True Negatives, FP = False Positives, FN = False Negatives, and TP = True Positives.

{{<definition "Classification metrics" classification_metrics>}}
**Accuracy** measures overall correctness: $\frac{TP + TN}{TP + TN + FP + FN}$

**Precision** answers "of predicted positives, how many are correct?": $\frac{TP}{TP + FP}$

**Recall (Sensitivity)** answers "of actual positives, how many did we detect?": $\frac{TP}{TP + FN}$

**Specificity** answers "of actual negatives, how many did we correctly identify?": $\frac{TN}{TN + FP}$

**F1-Score** balances precision and recall: $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
{{</definition>}}

The choice of metric depends critically on the problem context. Consider spam detection where 95% of emails are legitimate. A classifier that always predicts "not spam" achieves 95% accuracy but has zero recall—it catches no spam at all! In such imbalanced scenarios, precision and recall provide more meaningful evaluation. Medical diagnosis presents another perspective: high recall ensures we catch most disease cases (minimizing false negatives), while high precision reduces unnecessary treatments from false positives.

## The perceptron algorithm

### Model definition and geometry

The perceptron, introduced by Rosenblatt in 1958, provides our first glimpse into linear classification beyond the probabilistic framework of logistic regression. Using the $\\{-1, +1\\}$ label convention, the perceptron model is:

\begin{equation}
f(\mathbf{x}) = \text{sign}(\mathbf{w}^{\mathrm{T}} \mathbf{x} + b)
\label{eq:perceptron_model}
\end{equation}

where $\text{sign}(z) = +1$ if $z > 0$ and $-1$ otherwise.

The geometric interpretation reveals the elegance of this formulation. The decision boundary satisfies $\mathbf{w}^{\mathrm{T}} \mathbf{x} + b = 0$, defining a hyperplane in $\mathbb{R}^d$. The vector $\mathbf{w}$ serves as the normal to this hyperplane, pointing toward the positive class region. The bias $b$ controls the hyperplane's offset from the origin—specifically, the perpendicular distance from the origin to the hyperplane equals $\frac{|b|}{\|\mathbf{w}\|}$.

### The perceptron loss function

Unlike logistic regression's smooth cross-entropy loss, the perceptron employs a more direct approach: it only penalizes misclassified points. This leads to the perceptron loss:

\begin{equation}
L(\mathbf{w}, b) = \sum\_{i \in \mathcal{M}} -y_i(\mathbf{w}^{\mathrm{T}} \mathbf{x}_i + b)
\label{eq:perceptron_loss}
\end{equation}

where $\mathcal{M} = \\{i : y_i(\mathbf{w}^{\mathrm{T}} \mathbf{x}_i + b) \leq 0\\}$ denotes the set of misclassified points.

To understand this loss, consider a point $(\mathbf{x}_i, y_i)$:
- If correctly classified: $y_i(\mathbf{w}^{\mathrm{T}} \mathbf{x}_i + b) > 0$, contributing zero to the loss
- If misclassified: $y_i(\mathbf{w}^{\mathrm{T}} \mathbf{x}_i + b) < 0$, contributing $|y_i(\mathbf{w}^{\mathrm{T}} \mathbf{x}_i + b)|$ to the loss

The loss measures how "wrong" we are on misclassified points, with larger violations incurring greater penalties.

### Gradient computation and update rule

Computing gradients of the perceptron loss yields remarkably simple expressions:

\begin{equation}
\frac{\partial L}{\partial \mathbf{w}} = -\sum\_{i \in \mathcal{M}} y_i \mathbf{x}_i
\label{eq:perceptron_grad_w}
\end{equation}

\begin{equation}
\frac{\partial L}{\partial b} = -\sum\_{i \in \mathcal{M}} y_i
\label{eq:perceptron_grad_b}
\end{equation}

These gradients lead to the classic perceptron update rule. For each misclassified point $(\mathbf{x}_i, y_i)$:

\begin{equation}
\begin{aligned}
\mathbf{w} &\leftarrow \mathbf{w} + \alpha y_i \mathbf{x}_i \\\\
b &\leftarrow b + \alpha y_i
\end{aligned}
\label{eq:perceptron_update}
\end{equation}

The geometric intuition is compelling: we adjust the hyperplane to better classify the misclassified point by moving the normal vector $\mathbf{w}$ in the direction that reduces the violation.

### Introduction to stochastic optimization

The perceptron naturally motivates stochastic optimization. Consider a dataset with millions of points—computing the full gradient over all misclassified points becomes computationally prohibitive. This challenge extends beyond the perceptron to all large-scale machine learning.

**Full batch gradient descent** processes all data at each iteration:

\begin{equation}
\boldsymbol{\theta}^{(k+1)} = \boldsymbol{\theta}^{(k)} - \alpha \frac{1}{n} \sum\_{i=1}^n \nabla\_{\boldsymbol{\theta}} L_i(\boldsymbol{\theta}^{(k)})
\label{eq:batch_gd}
\end{equation}

with computational cost $O(n)$ per iteration.

**Stochastic gradient descent (SGD)** instead uses a single randomly selected example:

\begin{equation}
\boldsymbol{\theta}^{(k+1)} = \boldsymbol{\theta}^{(k)} - \alpha \nabla\_{\boldsymbol{\theta}} L_i(\boldsymbol{\theta}^{(k)})
\label{eq:sgd}
\end{equation}

where $i$ is sampled uniformly from $\\{1, 2, \ldots, n\\}$. The computational cost drops to $O(1)$ per iteration.

The key insight: while each SGD update is noisy, it provides an unbiased estimate of the true gradient. Mathematically, $\mathbb{E}[\nabla\_{\boldsymbol{\theta}} L_i(\boldsymbol{\theta})] = \frac{1}{n} \sum\_{j=1}^n \nabla\_{\boldsymbol{\theta}} L_j(\boldsymbol{\theta})$. Over many iterations, these noisy steps average out, leading to convergence (with appropriate learning rate decay).

**Mini-batch SGD** strikes a balance, using a small subset $B$ of examples:

\begin{equation}
\boldsymbol{\theta}^{(k+1)} = \boldsymbol{\theta}^{(k)} - \alpha \frac{1}{|B|} \sum\_{i \in B} \nabla\_{\boldsymbol{\theta}} L_i(\boldsymbol{\theta}^{(k)})
\label{eq:minibatch}
\end{equation}

Typical batch sizes range from 32 to 256, leveraging vectorized computation while maintaining reasonable variance in gradient estimates.

{{<theorem "Perceptron convergence theorem" perceptron_convergence>}}
If the training data is linearly separable with margin $\gamma > 0$ and all $\|\mathbf{x}_i\| \leq R$, then the perceptron algorithm converges in at most $(R/\gamma)^2$ iterations.
{{</theorem>}}

{{<proof>}}
The proof relies on two key observations: (1) each update increases $\mathbf{w}^{\mathrm{T}} \mathbf{w}^\star$ where $\mathbf{w}^\star$ is a separating hyperplane, and (2) $\|\mathbf{w}\|$ grows slowly. The ratio of these quantities bounds the number of updates. See Novikoff (1962) for the complete proof.
{{</proof>}}

## Support vector machines

### Motivation: which hyperplane is best?

The perceptron finds *a* separating hyperplane when data is linearly separable, but infinitely many such hyperplanes exist. Which should we prefer? Support vector machines answer this question elegantly: choose the hyperplane that maximizes the margin—the distance to the nearest data points.

This maximum margin principle embodies a form of regularization. By staying as far as possible from all training points, we build in robustness to small perturbations and improve generalization to unseen data. The connection to our optimization framework becomes clear: we'll formulate margin maximization as a constrained optimization problem.

### Mathematical formulation

Consider a linearly separable dataset with labels $y_i \in \\{-1, +1\\}$. Any separating hyperplane satisfies $y_i(\mathbf{w}^{\mathrm{T}} \mathbf{x}_i + b) > 0$ for all $i$. The SVM introduces a crucial normalization: we scale $\mathbf{w}$ and $b$ such that the closest points to the hyperplane satisfy:

\begin{equation}
y_i(\mathbf{w}^{\mathrm{T}} \mathbf{x}_i + b) = 1
\label{eq:svm_normalization}
\end{equation}

This canonical form ensures all points satisfy:

\begin{equation}
y_i(\mathbf{w}^{\mathrm{T}} \mathbf{x}_i + b) \geq 1 \quad \forall i = 1, \ldots, n
\label{eq:svm_constraints}
\end{equation}

The geometric distance from a point $\mathbf{x}_0$ to the hyperplane $\mathbf{w}^{\mathrm{T}} \mathbf{x} + b = 0$ equals:

$$\text{distance} = \frac{|\mathbf{w}^{\mathrm{T}} \mathbf{x}_0 + b|}{\|\mathbf{w}\|}$$

For the support vectors (points achieving equality in \eqref{eq:svm_constraints}), this distance equals $\frac{1}{\|\mathbf{w}\|}$. The margin—the total separation between the two classes—therefore equals $\frac{2}{\|\mathbf{w}\|}$.

### The primal optimization problem

Maximizing the margin $\frac{2}{\|\mathbf{w}\|}$ is equivalent to minimizing $\|\mathbf{w}\|$. For mathematical convenience, we minimize $\frac{1}{2}\|\mathbf{w}\|^2$, leading to the SVM primal problem:

\begin{equation}
\begin{aligned}
\min\_{\mathbf{w}, b} \quad & \frac{1}{2} \|\mathbf{w}\|^2 \\\\
\text{subject to} \quad & y_i(\mathbf{w}^{\mathrm{T}} \mathbf{x}_i + b) \geq 1, \quad i = 1, \ldots, n
\end{aligned}
\label{eq:svm_primal}
\end{equation}

This is a convex quadratic program—the objective is a positive definite quadratic function, and the constraints are linear. Our optimization theory guarantees a unique global minimum.

### Lagrangian formulation and KKT conditions

Introducing Lagrange multipliers $\alpha_i \geq 0$ for each constraint, we form the Lagrangian:

\begin{equation}
\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}\|\mathbf{w}\|^2 - \sum\_{i=1}^n \alpha_i [y_i(\mathbf{w}^{\mathrm{T}} \mathbf{x}_i + b) - 1]
\label{eq:svm_lagrangian}
\end{equation}

The KKT conditions for this problem are:

**Stationarity:**

\begin{equation}
\nabla\_{\mathbf{w}} \mathcal{L} = \mathbf{w} - \sum\_{i=1}^n \alpha_i y_i \mathbf{x}_i = 0
\label{eq:kkt_stationarity_w}
\end{equation}

\begin{equation}
\nabla_b \mathcal{L} = -\sum\_{i=1}^n \alpha_i y_i = 0
\label{eq:kkt_stationarity_b}
\end{equation}

**Primal feasibility:**
$$y_i(\mathbf{w}^{\mathrm{T}} \mathbf{x}_i + b) - 1 \geq 0$$

**Dual feasibility:**
$$\alpha_i \geq 0$$

**Complementary slackness:**
\begin{equation}
\alpha_i [y_i(\mathbf{w}^{\mathrm{T}} \mathbf{x}_i + b) - 1] = 0
\label{eq:complementary_slackness}
\end{equation}

The complementary slackness condition reveals the sparse nature of the SVM solution. For each point, either $\alpha_i = 0$ (the point doesn't influence the solution) or $y_i(\mathbf{w}^{\mathrm{T}} \mathbf{x}_i + b) = 1$ (the point lies exactly on the margin boundary). Points with $\alpha_i > 0$ are called **support vectors**—they alone determine the hyperplane.

### The dual problem

From the stationarity condition \eqref{eq:kkt_stationarity_w}, we have:

$$\mathbf{w} = \sum\_{i=1}^n \alpha_i y_i \mathbf{x}_i$$

Substituting this and the constraint $\sum\_{i=1}^n \alpha_i y_i = 0$ back into the Lagrangian yields the dual objective:

\begin{equation}
\mathcal{L}_D(\boldsymbol{\alpha}) = \sum\_{i=1}^n \alpha_i - \frac{1}{2} \sum\_{i=1}^n \sum\_{j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_i^{\mathrm{T}} \mathbf{x}_j
\label{eq:svm_dual_objective}
\end{equation}

The SVM dual problem becomes:

\begin{equation}
\begin{aligned}
\max\_{\boldsymbol{\alpha}} \quad & \sum\_{i=1}^n \alpha_i - \frac{1}{2} \sum\_{i=1}^n \sum\_{j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_i^{\mathrm{T}} \mathbf{x}_j \\\\
\text{subject to} \quad & \sum\_{i=1}^n \alpha_i y_i = 0 \\\\
& \alpha_i \geq 0, \quad i = 1, \ldots, n
\end{aligned}
\label{eq:svm_dual}
\end{equation}

This dual formulation offers several advantages. The problem size depends on the number of training examples $n$, not the dimension $d$. More importantly, the objective depends on the data only through inner products $\mathbf{x}_i^{\mathrm{T}} \mathbf{x}_j$—this observation leads to the kernel trick for nonlinear classification.

### Computing the final classifier

Once we solve for the optimal $\boldsymbol{\alpha}^\star$, we reconstruct the primal variables:

$$\mathbf{w}^\star = \sum\_{i=1}^n \alpha_i^\star y_i \mathbf{x}_i$$

To find $b^\star$, we use any support vector $\mathbf{x}_s$ (where $\alpha_s^\star > 0$). From complementary slackness, $y_s(\mathbf{w}^{*\mathrm{T}} \mathbf{x}_s + b^\star) = 1$, giving:

\begin{equation}
b^\star = y_s - \mathbf{w}^{*\mathrm{T}} \mathbf{x}_s = y_s - \sum\_{i=1}^n \alpha_i^\star y_i \mathbf{x}_i^{\mathrm{T}} \mathbf{x}_s
\label{eq:svm_compute_b}
\end{equation}

The final classification function is:

\begin{equation}
f(\mathbf{x}) = \text{sign}\left(\sum\_{i=1}^n \alpha_i^\star y_i \mathbf{x}_i^{\mathrm{T}} \mathbf{x} + b^\star\right)
\label{eq:svm_classifier}
\end{equation}

Remarkably, only support vectors contribute to this sum—most $\alpha_i^\star$ equal zero. This sparsity makes SVMs computationally efficient at test time and provides insight into which training examples are most informative.

## Cross-validation

### K-fold cross-validation

Machine learning models contain hyperparameters that cannot be learned through optimization—for instance, the learning rate in gradient descent or the regularization strength in SVMs. Cross-validation provides a principled approach to hyperparameter selection and performance estimation.

{{<definition "K-fold cross-validation" kfold_cv>}}
**K-fold cross-validation** partitions the training data into $K$ equal-sized folds. For each fold $k$:
1. Train the model on folds $\\{1, 2, \ldots, K\\} \setminus \\{k\\}$
2. Evaluate on fold $k$
3. Record the validation performance

The final performance estimate is the average across all $K$ folds.
{{</definition>}}

The algorithm proceeds as follows:
1. Randomly shuffle the dataset
2. Split into $K$ approximately equal folds
3. For $k = 1$ to $K$:
   - Combine folds $\\{1, \ldots, K\\} \setminus \\{k\\}$ as training set
   - Use fold $k$ as validation set
   - Train model and compute validation metric
4. Return mean and standard deviation of $K$ validation scores

Common choices include $K = 5$ or $K = 10$, balancing computational cost against variance in the estimate. The extreme case $K = n$ (leave-one-out cross-validation) provides an nearly unbiased estimate but becomes computationally prohibitive for large datasets.

### Stratified cross-validation

For classification problems, especially with imbalanced classes, **stratified K-fold** maintains the class distribution in each fold. If 10% of examples are positive in the full dataset, each fold should contain approximately 10% positive examples. This prevents pessimistic estimates from folds that lack examples from minority classes.

### Applications of cross-validation

Cross-validation serves three primary purposes in machine learning:

**Model selection** compares different algorithms (e.g., logistic regression vs. SVM) using their cross-validated performance. The model with the best average validation score is selected for final training on the full dataset.

**Hyperparameter tuning** searches over hyperparameter values (e.g., regularization strength $C$ in SVM) to find settings that maximize cross-validated performance. Grid search exhaustively tries combinations, while random search samples from distributions over hyperparameters.

**Performance estimation** provides a robust estimate of how the model will perform on unseen data. The standard deviation across folds indicates the stability of this estimate.

## Summary and exercises

We've extended our optimization framework to classification, introducing two fundamental algorithms. The perceptron uses a simple loss that only penalizes misclassified points, naturally leading to stochastic gradient descent. Support vector machines formulate classification as constrained optimization, finding the maximum margin hyperplane through the elegant machinery of Lagrangian duality.

The key insights connecting to our optimization foundations:
- Classification losses differ from regression but still yield optimization problems
- Stochastic gradient descent addresses computational challenges of large datasets
- Constrained optimization and KKT conditions lead to powerful algorithms like SVM
- The dual formulation reveals computational advantages and enables extensions

### Exercises

1. **Perceptron implementation**: Implement the perceptron algorithm from scratch. Test on a 2D dataset where you can visualize the decision boundary evolution during training. Compare batch versus stochastic updates.

2. **SVM with sklearn**: Apply SVM to a subset of MNIST digits (e.g., 3 vs. 8). First use PCA to reduce dimensionality to 50 components, then tune the regularization parameter $C$ using 5-fold cross-validation. Report test accuracy and number of support vectors.

3. **Derive the soft-margin SVM**: Extend the hard-margin SVM to handle non-separable data by introducing slack variables $\xi_i \geq 0$. Write the primal problem, derive the KKT conditions, and show how the dual problem changes.
