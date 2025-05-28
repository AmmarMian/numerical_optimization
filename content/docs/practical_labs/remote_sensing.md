---
title: II - Remote Sensing project
weight: 20
---

# Solving Inverse Problems in Remote Sensing 

## Introduction

### What is a Hyperspectral Image?

A **hyperspectral image** captures information across a wide range of the electromagnetic spectrum. Unlike traditional images that capture data in three bands (red, green, and blue), hyperspectral images can capture data in hundreds of contiguous spectral bands. This allows for detailed spectral analysis and identification of materials in **remote sensing** based on their spectral signatures.

{{< center >}}
{{< NumberedFigure
  src="../../../../tikZ/himmelblau/main_2D.svg"
  caption="An example of a Hyperspectral image"
  alt="HSI image"
  width="500px"
  label="fig:hsi"
>}}
{{< /NumberedFigure >}}

**1.** *Clone the git repo for the lab session and open the show_hsi.ipynb notebook. You can use the script to load an image from the HSI dataset and visualize the various band images.*

**2.** *Show the image in RGB by chosing the appropriate image bands to form a 3D image (take a look at the README.md).*

### Spectral Unmixing Linear Model

Mostly used in remote sensing, hyperspectral images of the Earth are composed of pixels that represent mixed spectral signatures of various materials. The spectra observed at each pixel is the result of multiple constituent spectra called *endmembers*.

It is commonly accepted to model the pixel spectra by a linear mixing model as follows:

$$ 
\mathbf{y}\_p = \sum_{k=1}^K a_{kp} \mathbf{s}_k + \mathbf{n}_p 
$$

where:
- $ \mathbf{y}_p \in \mathbb{R}^m $ is the observed spectrum at pixel $ p $,
- $ a_{kp} \in \mathbb{R} $ is the abundance of the $ k $-th endmember at pixel $ p $,
- $ \mathbf{s}_k \in \mathbb{R}^m $ is the spectral signature of the $ k $-th endmember,
- $ \mathbf{n}_p \in \mathbb{R}^m $ represents the noise at pixel $ p $.

Spectral unmixing is the process of decomposing a mixed pixel in a hyperspectral image into a set of endmembers and their corresponding abundances. The linear mixing model can be written in matrix form as:

$$ \mathbf{Y} = \mathbf{A} \mathbf{S} + \mathbf{N} $$

where:
- $ \mathbf{Y} \in \mathbb{R}^{m \times n} $ is the matrix of observed spectra,
- $ \mathbf{A} \in \mathbb{R}^{n \times K} $ is the matrix of abundances,
- $ \mathbf{S} \in \mathbb{R}^{K \times m} $ is the matrix of endmember spectra,
- $ \mathbf{N} \in \mathbb{R}^{m \times n} $ represents the noise matrix.



### Formulation of the Inverse Problem

An inverse problem involves determining the input or parameters of a system from its observed output. In the context of spectral unmixing, the inverse problem is to estimate the endmember spectra and their abundances from the observed hyperspectral data.

In the context of spectral unmixing, this means factorizing the observed data matrix $ \mathbf{Y} $ into an endmember matrix $ \mathbf{S} $ and an abundance matrix $ \mathbf{A} $. This can be formulated as a **least square optimization problem** where the goal is to minimize the difference between the observed data and the reconstructed data.

**3.** *Propose an objective function for this optimization problem.*

**4.** *Is the objective function convex ?*

**5.** *Can you find a closed form solution for $\mathbf{A}$ given $\mathbf{S}$ and inversely ?*

<!-- The objective function for this optimization problem can be written as:

$$ \min_{\mathbf{A}, \mathbf{S}} \Vert\mathbf{A} \mathbf{S} - \mathbf{Y}\Vert_F^2 $$ -->

<!-- where $ \Vert\mathbf{A} \mathbf{S} - \mathbf{Y}\Vert_F^2 $ is the Frobenius norm of the difference between the reconstructed data $ \mathbf{A} \mathbf{S} $ and the observed data $ \mathbf{Y} $. -->


## Part 1 - Solving the least square inverse problem


### The Block-Coordinate Descent algorithm

The Block-coordinate descent is an optimization algorithm that updates a subset of variables at each iteration, which is particularly useful for problems with separable structure. In the context of spectral unmixing, this method can be used to solve the optimization for $ \mathbf{A} $ and $ \mathbf{S} $ simultaneously.

$$ \min_{\mathbf{A}, \mathbf{S}} F(\mathbf{A}, \mathbf{S}) $$

The intuition behind block-coordinate descent is to break down a complex bi-level optimization problem into simpler subproblems that can be solved more efficiently. The algorithm alternated between the subproblems by fixing one set of variables and optimizing the other, cycling through all blocks until convergence.
The algorithm iterates as follows:

1. **Initialize** $ \mathbf{A} $ and $ \mathbf{S} $.
2. **Repeat until convergence**:
   - Fix $ \mathbf{S} $ and update $ \mathbf{A} $ by solving:
     $$ \min_{\mathbf{A}} F(\mathbf{A}, \mathbf{S}) $$
   - Fix $ \mathbf{A} $ and update $ \mathbf{S} $ by solving:
     $$ \min_{\mathbf{S}} F(\mathbf{A}, \mathbf{S}) $$



The convergence of block-coordinate descent is guaranteed under certain conditions, such as the convexity of the objective function and the proper choice of step sizes.


**6.** *By using the expression obtained at (**5.**), propose and implement a Block-coordinate descent scheme to solve for $ \mathbf{A} $ and $ \mathbf{S} $*

<!-- The algorithm iterates as follows:

1. **Initialize** $ \mathbf{A} $ and $ \mathbf{S} $.
2. **Repeat until convergence**:
   - Fix $ \mathbf{S} $ and update $ \mathbf{A} $ by solving:
     $$ \min_{\mathbf{A}} \Vert\mathbf{A} \mathbf{S} - \mathbf{Y}\Vert_F^2 + \lambda \Vert\mathbf{A}\Vert_F^2 $$
   - Fix $ \mathbf{A} $ and update $ \mathbf{S} $ by solving:
     $$ \min_{\mathbf{S}} \Vert\mathbf{A} \mathbf{S} - \mathbf{Y}\Vert_F^2 $$ -->


**7.** Measure the time and plot evolution of cost with iterations...

**8.** Derive a descent algorithm to solve the sub-problems

**9.** Compare the total computation cost (in cpu time)...

**10.** Compute the dimension of the solution space for least square...

**11.** *Can you comment on the quality of the results obtained ?*


## Part 2 - Ill-Posed Problems and Regularization

Inverse problems, such as spectral unmixing, are often ill-posed, meaning that they may not have a unique solution or the solution may be highly sensitive to noise.
Thus, regularization techniques are used to stabilize the solution and reduce the impact of noise by restricting the solution space. They can be implemented as constraints in the optimization problem or relaxed to allow some flexibility. 




Ridge regression is one such technique that introduces a regularization term to the objective function:

$$ \min_{\mathbf{A}, \mathbf{S}} \Vert\mathbf{A} \mathbf{S} - \mathbf{Y}\Vert_F^2 + \lambda \Vert\mathbf{S}\Vert_F^2 $$

where $ \lambda \in \mathbb{R} $ is the regularization parameter that controls the trade-off between fitting the data and keeping the solution simple.


## Part 3 (Bonus) - Apply to Audio Source Separation


<!-- 

### Can We Find a Closed Form Solution?

Consider the following questions to engage with the concept of closed-form solutions:

1. **Theoretical Question**: Under what conditions does a closed-form solution exist for the ridge regression problem?
2. **Practical Question**: How might the size of the dataset influence the feasibility of using a closed-form solution?
3. **Implementation Question**: What are the computational challenges of inverting a large matrix in Python?

For certain optimization problems, it is possible to find a closed-form solution that directly gives the optimal values of the variables. In the context of ridge regression, the closed-form solution for the abundance matrix $ \mathbf{A} $ can be derived as:

$$ \mathbf{A} = (\mathbf{S}^T \mathbf{S} + \lambda \mathbf{I})^{-1} \mathbf{S}^T \mathbf{Y} $$

This solution involves inverting a matrix, which can be computationally intensive and may not be feasible for large datasets.

### Data Size Matters for Inversion

The computational complexity of matrix inversion depends on the size of the matrix. For large datasets, inverting the matrix $ \mathbf{S}^T \mathbf{S} + \lambda \mathbf{I} $ can be computationally expensive and may require significant memory and processing power. It is important to consider the size of the data when choosing between closed-form solutions and iterative methods.

1. **Theoretical Question**: How does the size of the matrix affect the computational complexity of inversion?
2. **Practical Question**: What are the memory and processing implications of inverting large matrices?
3. **Implementation Question**: How can you optimize matrix inversion in Python for large datasets?

## Part 2 - Gradient Descent

### Introduction

Gradient descent is an iterative optimization algorithm used to minimize the objective function. This section will explore how to compute the gradient, implement the gradient descent algorithm, measure its performance, and optimize the step size.

### Compute the Gradient

Gradient descent is an iterative optimization algorithm used to minimize the objective function. The gradient of the objective function with respect to the variables is computed and used to update the variables in the direction of the steepest descent.

1. **Theoretical Question**: What is the role of the gradient in the optimization process?
2. **Practical Question**: How do you compute the gradient for a given objective function in Python?
3. **Implementation Question**: What are the challenges in implementing gradient descent for large datasets?

The gradient of the objective function for ridge regression is given by:

$$ \nabla_{\mathbf{A}} \Vert\mathbf{A} \mathbf{S} - \mathbf{Y}\Vert_F^2 + \lambda \Vert\mathbf{A}\Vert_F^2 = 2 (\mathbf{A} \mathbf{S} - \mathbf{Y}) \mathbf{S}^T + 2 \lambda \mathbf{A} $$

### Implement a Gradient Descent

To implement gradient descent, we start with an initial guess for the variables and iteratively update them using the gradient. The update rule for gradient descent is:

$$ \mathbf{A}_{k+1} = \mathbf{A}_k - \alpha \nabla_{\mathbf{A}} f(\mathbf{A}_k) $$

where $ \alpha \in \mathbb{R} $ is the step size, and $ \nabla_{\mathbf{A}} f(\mathbf{A}_k) $ is the gradient of the objective function at $ \mathbf{A}_k $.

1. **Theoretical Question**: How does the choice of initial guess affect the convergence of gradient descent?
2. **Practical Question**: What are the considerations for choosing the step size $ \alpha $?
3. **Implementation Question**: How can you ensure the gradient descent algorithm converges to a stable solution in Python?

### Measure Performances

The performance of the gradient descent algorithm can be measured using metrics such as the mean squared error (MSE) between the true and estimated abundances. It is important to monitor the convergence of the algorithm and ensure that it reaches a stable solution.

1. **Theoretical Question**: What metrics can be used to measure the performance of gradient descent?
2. **Practical Question**: How do you monitor the convergence of the algorithm in practice?
3. **Implementation Question**: How can you visualize the convergence of the gradient descent algorithm in Python?

### Optimize Step Size

The step size $ \alpha $ plays a crucial role in the convergence of the gradient descent algorithm. A step size that is too large can cause the algorithm to diverge, while a step size that is too small can result in slow convergence. Techniques such as line search can be used to optimize the step size and improve the performance of the algorithm.

1. **Theoretical Question**: Why is the step size important in gradient descent?
2. **Practical Question**: How can line search be used to optimize the step size?
3. **Implementation Question**: How do you implement line search in Python to optimize the step size?

## Part 3 - Newton Algorithm

### Introduction

Newton's method is an iterative optimization algorithm that uses second-order derivative information to update the variables. This section will explore how to compute the Hessian, implement Newton's method, measure its performance, and optimize the step size.

### Compute the Hessian

Newton's method is an iterative optimization algorithm that uses the second-order derivative information to update the variables. The Hessian of the objective function is computed and used to update the variables in the direction of the Newton step.

1. **Theoretical Question**: What is the role of the Hessian in Newton's method?
2. **Practical Question**: How do you compute the Hessian for a given objective function in Python?
3. **Implementation Question**: What are the challenges in implementing Newton's method for large datasets?

The Hessian of the objective function for ridge regression is given by:

$$ \nabla^2_{\mathbf{A}} \Vert\mathbf{A} \mathbf{S} - \mathbf{Y}\Vert_F^2 + \lambda \Vert\mathbf{A}\Vert_F^2 = 2 \mathbf{S} \mathbf{S}^T + 2 \lambda \mathbf{I} $$

### Implement Newton's Method

To implement Newton's method, we start with an initial guess for the variables and iteratively update them using the Hessian. The update rule for Newton's method is:

$$ \mathbf{A}_{k+1} = \mathbf{A}_k - (\nabla^2_{\mathbf{A}} f(\mathbf{A}_k))^{-1} \nabla_{\mathbf{A}} f(\mathbf{A}_k) $$

where $ \nabla^2_{\mathbf{A}} f(\mathbf{A}_k) $ is the Hessian of the objective function at $ \mathbf{A}_k $, and $ \nabla_{\mathbf{A}} f(\mathbf{A}_k) $ is the gradient of the objective function at $ \mathbf{A}_k $.

1. **Theoretical Question**: How does the choice of initial guess affect the convergence of Newton's method?
2. **Practical Question**: What are the considerations for using the Hessian in Newton's method?
3. **Implementation Question**: How can you ensure Newton's method converges to a stable solution in Python?

### Measure Performances

The performance of Newton's method can be measured using metrics such as the mean squared error (MSE) between the true and estimated abundances. It is important to monitor the convergence of the algorithm and ensure that it reaches a stable solution.

1. **Theoretical Question**: What metrics can be used to measure the performance of Newton's method?
2. **Practical Question**: How do you monitor the convergence of the algorithm in practice?
3. **Implementation Question**: How can you visualize the convergence of Newton's method in Python?

### Optimize Step Size

The step size in Newton's method is determined by the Hessian, which provides information about the curvature of the objective function. Techniques such as line search can be used to optimize the step size and improve the performance of the algorithm.

1. **Theoretical Question**: Why is the step size important in Newton's method?
2. **Practical Question**: How can line search be used to optimize the step size in Newton's method?
3. **Implementation Question**: How do you implement line search in Python to optimize the step size in Newton's method?


### Introduction to Audio Source Separation

Audio source separation is the process of decomposing a mixed audio signal into its constituent sources.  
The STFT is a representation blabla... source separation can be viewed as matrix factorization...

You can find test audio samples on data folder and a script to compute the STFT... 

*Implement audio source separation from what's been done below...*   -->
