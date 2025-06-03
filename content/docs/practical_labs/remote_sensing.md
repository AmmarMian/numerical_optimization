---
title: II - Remote Sensing Project
weight: 20
---

# Solving Inverse Problems in Remote Sensing

## README

- [Where to find the code](../code/remote_sensing_code.py)
- [Content of the repo](../data/indian_pines/)
- [Where to find the data](../data/indian_pines/Indian_pines_corrected.mat)

## Introduction

In this lab session, we will explore one of the many applications of numerical optimization, namely **solving inverse problems**. Inverse problems constitute a sub-field of applied mathematics with many applications in real-world data analysis. In this lab, you will work with **hyperspectral images**, commonly used in **remote sensing**.

We will begin with an introduction to hyperspectral images and derive a formulation of the **hyperspectral unmixing problem**. We will then express the inverse problem through an objective function to optimize. In the second part of the lab, we will experiment with multiple algorithms to solve the optimization problem and analyze their performance.

Hyperspectral unmixing is a fundamental problem in remote sensing that involves decomposing mixed pixel spectra into their constituent materials (endmembers) and their respective proportions (abundances). This problem is inherently an inverse problem where we seek to recover the underlying components from observed mixed signals.

As a bonus, we will explore blind hyperspectral unmixing where both endmembers and abundances are unknown, making the problem significantly more challenging.

## Learning Objectives

By the end of the session, you should be able to:

- Derive a data model and objective function to solve an inverse problem
- Implement descent algorithms to solve multi-objective optimization problems
- Handle constrained optimization problems using Lagrange multipliers and projection methods
- Benchmark optimization algorithms and measure their performance
- Manipulate real-world hyperspectral data in Python
- Understand the relationship between physical constraints and mathematical optimization

## I - Modelization and Problem Setup

### 1. What is a Hyperspectral Image?

A **hyperspectral image (HSI)** captures information across a wide range of the electromagnetic spectrum. Unlike traditional images that capture data in three bands (red, green, and blue), hyperspectral images can capture data in hundreds of contiguous spectral bands. This allows for detailed spectral analysis and identification of materials in **remote sensing** based on their spectral signatures.

Each pixel in a hyperspectral image contains a complete spectrum, which can be thought of as a "fingerprint" of the materials present in that pixel. The spectral dimension provides rich information about the chemical and physical properties of the observed scene.

{{<center>}}
{{<NumberedFigure
  src="../../../../tikZ/hyperspectral/hsi_cube.svg"
  alt="Hyperspectral image cube showing spatial and spectral dimensions"
  width="500px"
  caption="A hyperspectral image cube with spatial dimensions (x,y) and spectral dimension (λ)"
  label="fig:hsi_cube"
>}}
{{</center>}}

In this lab session, we will work with an open dataset for hyperspectral data analysis: the **Indian Pines HSI dataset**. This dataset is widely used in the remote sensing community and contains agricultural fields with different crop types.

**Tasks:**

1. **Open the Indian Pines dataset using the `loadmat` function from scipy** ([documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html)).

2. **What is the size of the image and how many spectral bands does the image contain?**

{{<hiddenhint "Hint">}}
The Indian Pines dataset typically has dimensions of 145×145 pixels with 200 spectral bands after noise removal. You can check the shape using `.shape` attribute in Python.
{{</hiddenhint>}}

3. **Use `imshow` to display a few band images of the HSI cube at different wavelengths.** Try displaying bands at different spectral regions (e.g., visible, near-infrared, short-wave infrared).

4. **Load and display the ground truth classification map.** This shows the different crop types present in the scene.

5. **Extract and plot the mean spectrum of the first three classes** from the ground truth. What differences do you observe between the spectral signatures?

{{<hiddenhint "Hint">}}
Use the ground truth labels to mask the hyperspectral data and compute the mean spectrum for each class. Different materials will have distinct spectral signatures, particularly in the near-infrared region.
{{</hiddenhint>}}

### 2. The Spectral Unmixing Linear Model

In remote sensing, hyperspectral images of the Earth are composed of pixels that represent mixed spectral signatures of various materials. Due to the limited spatial resolution of sensors, each pixel often contains multiple materials. The spectrum observed at each pixel is the result of multiple constituent spectra called **endmembers**. 

**Spectral unmixing** consists of estimating the per-pixel **abundances** of each endmember, giving insight into the various constituents of the observed field. This is particularly important in applications such as:
- Agricultural monitoring (crop type identification)
- Environmental monitoring (vegetation health assessment)
- Geological surveys (mineral identification)
- Urban planning (land cover classification)

{{<center>}}
{{<NumberedFigure
  src="../../../../tikZ/hyperspectral/spectral_unmixing.svg"
  alt="Illustration of spectral unmixing process"
  width="600px"
  caption="Spectral unmixing: decomposing mixed pixel spectra into endmember spectra and abundances"
  label="fig:unmixing"
>}}
{{</center>}}

It is commonly accepted to model the pixel spectra by a **linear mixing model** as follows:

$$
\mathbf{y}_p = \sum\_{k=1}^K a\_{kp} \mathbf{s}_k + \mathbf{n}_p
$$

where:
- $\mathbf{y}_p \in \mathbb{R}^m$ is the observed spectrum at pixel $p$ (with $m$ spectral bands)
- $a\_{kp} \in \mathbb{R}$ is the abundance (proportion) of the $k$-th endmember at pixel $p$
- $\mathbf{s}_k \in \mathbb{R}^m$ is the spectral signature of the $k$-th endmember
- $\mathbf{n}_p \in \mathbb{R}^m$ represents the noise at pixel $p$
- $K$ is the number of endmembers

The linear mixing model assumes that (i) the observed spectrum is a linear combination of endmember spectra, (ii) there are no multiple scattering effects and (iii) the endmembers are spectrally distinct.

**Tasks:**

1. **Derive a matrix formulation of the linear mixing model** in which images are vectorized. Define clearly:
   - The data matrix $\mathbf{Y} \in \mathbb{R}^{m \times n}$ where $n$ is the number of pixels
   - The endmember matrix $\mathbf{S} \in \mathbb{R}^{m \times K}$
   - The abundance matrix $\mathbf{A} \in \mathbb{R}^{K \times n}$
   - The noise matrix $\mathbf{N} \in \mathbb{R}^{m \times n}$

{{<hiddenhint "Hint">}}
Vectorize the spatial dimension by stacking the pixels column-wise.
{{</hiddenhint>}}

2. **What are the physical constraints** that should be imposed on the abundance matrix $\mathbf{A}$? Justify your answer.

{{<hiddenhint "Hint">}}
Think about what abundances represent physically. They are proportions of materials in a pixel. Remember that we assume that the endmember matrix includes all the material present in the scene. 
{{</hiddenhint>}}

### 3. Formulation of the Inverse Problem

The **inverse problem** in hyperspectral unmixing consists of estimating the endmember matrix $\mathbf{S}$ and the abundance matrix $\mathbf{A}$ from the observed data matrix $\mathbf{Y}$. This is essentially a **matrix factorization problem**.

The basic objective function for this optimization problem can be written as:

$$
\min\_{\mathbf{A}, \mathbf{S}} \Vert\mathbf{S}\mathbf{A} - \mathbf{Y}\Vert_F^2
$$

where $\Vert\cdot\Vert_F^2$ is the squared Frobenius norm of the matrix, defined as:
$$
\Vert\mathbf{X}\Vert_F^2 = \sum\_{i,j} x\_{ij}^2 = \text{tr}(\mathbf{X}^T\mathbf{X})
$$

**Tasks:**

1. **Explain why this is called an inverse problem.** What makes it challenging compared to a forward problem?

2. **Is the objective function convex in both $\mathbf{A}$ and $\mathbf{S}$ simultaneously?** Justify your answer.

{{<hiddenhint "Hint">}}
Is it convex in $\mathbf{A}$ when $\mathbf{S}$ is fixed, and convex in $\mathbf{S}$ when $\mathbf{A}$ is fixed ? Is it jointly convex in both variables ? 
{{</hiddenhint>}}

## II - Solving the Unconstrained Least Squares Inverse Problem

An inverse problem involves determining the input or parameters of a system from its observed output. In the context of spectral unmixing, the inverse problem is to estimate the endmember spectra and their abundances from the observed hyperspectral data.

**We will mostly consider the case where the endmembers $\mathbf{S}$ are known** (e.g., from a spectral library or field measurements). In this case, we only need to estimate the abundance matrix $\mathbf{A}$.

### 1. Unconstrained Least Squares Solution

When the endmembers are known, the problem becomes a **linear least squares problem** for each pixel:

$$
\min\_{\mathbf{A}} \Vert\mathbf{S}\mathbf{A} - \mathbf{Y}\Vert_F^2
$$

**Tasks:**

1. **Derive the analytical solution** for the unconstrained least squares problem. Show that the optimal abundance matrix is given by:
$$
\mathbf{A}^* = (\mathbf{S}^T\mathbf{S})^{-1}\mathbf{S}^T\mathbf{Y}
$$

{{<hiddenhint "Hint">}}
This is similar to the multiple linear regression problem from Lab I. Use the fact that the Frobenius norm can be expressed as a sum of vector norms, and solve for each pixel independently.
{{</hiddenhint>}}

2. **Under what conditions is this solution unique?** What happens if $\mathbf{S}^T\mathbf{S}$ is not invertible? 

3. **Implement the unconstrained least squares solution** as a lambda function or regular function.

<!-- 4. **For testing purposes, generate synthetic data** using known endmembers and abundances, then verify that your solution recovers the true abundances. -->

### 2. Performance Evaluation

To evaluate the quality of spectral unmixing results, we need appropriate metrics that measure both spectral and spatial accuracy.

**Tasks:**

1. **Implement a function that evaluates the following evaluation metrics:**
   - **Spectral Angle Mapper (SAM)**: Measures the angle between reconstructed and original spectra.
   - **Root Mean Square Error (RMSE)**: Measures the pixel-wise reconstruction error.
   - **SSIM** : Measure a perceptual similiarity between images.

Take a look at the documentation of scipy to find ..

2. **Implement a reconstruction and visualization function** that:
   - Reconstructs the hyperspectral image from estimated abundances
   - Displays RGB composite images (original vs. reconstructed)
   - Shows abundance maps for each endmember
   - Computes and displays the evaluation metrics

3. **Apply your unconstrained least squares solution** to the Indian Pines dataset:
   - Use the mean spectra of the first 3-5 classes as endmembers
   - Compute the abundance maps
   - Evaluate the reconstruction quality

4. **Comment on the results.** What do you observe about the abundance values? Are they physically meaningful?

{{<hiddenhint "Hint">}}
You'll likely observe that some abundance values are negative or that abundances for a pixel don't sum to one. This violates the physical constraints and motivates the need for constrained optimization.
{{</hiddenhint>}}

## III - Solving Constrained Least Squares Inverse Problems

Given the physical interpretation of the spectral linear mixing model, a set of constraints should be added to the optimization problem to ensure physically meaningful solutions.

**Tasks:**

1. **Propose a set of equality and/or inequality constraints** to ensure the interpretability of the solutions. Explain the physical meaning of each constraint.

{{<hiddenhint "Hint">}}
The two main physical constraints are:
- **Sum-to-one constraint**: $\sum\_{k=1}^K a\_{kp} = 1$ for all pixels $p$ (abundances are proportions)
- **Non-negativity constraint**: $a\_{kp} \geq 0$ for all $k, p$ (negative abundances are not physical)
{{</hiddenhint>}}

As you must notice, this makes the optimization problem more difficult to solve. In the next parts of the lab, we will derive methods to solve the relaxed version of the fully constrained problem.

### 1. Sum-to-One Constrained Least Squares

Let's first consider only the sum-to-one constraint. The optimization problem becomes:

$$
\begin{align}
\min\_{\mathbf{A}} &\quad \Vert\mathbf{S}\mathbf{A} - \mathbf{Y}\Vert_F^2 \\
\text{subject to} &\quad \mathbf{1}^T\mathbf{A} = \mathbf{1}^T
\end{align}
$$

where $\mathbf{1}$ is a vector of ones.

**Tasks:**

1. **Derive the Lagrangian** of the constrained optimization problem. For a single pixel $p$, the Lagrangian is:
$$
L(\mathbf{a}_p, \lambda_p) = \Vert\mathbf{S}\mathbf{a}_p - \mathbf{y}_p\Vert_2^2 + \lambda_p(1 - \mathbf{1}^T\mathbf{a}_p)
$$

2. **Compute the optimal solution** by setting the gradients to zero. Show that the solution is:
$$
\mathbf{a}_p^* = (\mathbf{S}^T\mathbf{S})^{-1}\mathbf{S}^T\mathbf{y}_p + (\mathbf{S}^T\mathbf{S})^{-1}\mathbf{1}\lambda_p^*
$$
where $\lambda_p^*$ is chosen to satisfy the constraint.
<!-- 
{{<hiddenhint "Hint">}}
Use the method of Lagrange multipliers. Set $\nabla\_{\mathbf{a}_p} L = 0$ and $\nabla\_{\lambda_p} L = 0$, then solve for $\lambda_p^*$ using the constraint.
{{</hiddenhint>}} -->

3. **Derive the closed-form expression** for $\lambda_p^*$ and the final solution.

4. **Implement the sum-to-one constrained solution** and test it on the Indian Pines dataset.

5. **Display the results**: RGB composite, abundance maps, and compute evaluation metrics. **Comment on the improvements** compared to the unconstrained solution.

### 2. Non-Negativity Constrained Least Squares

Now consider only the non-negativity constraints:

$$
\begin{align}
\min\_{\mathbf{A}} &\quad \Vert\mathbf{S}\mathbf{A} - \mathbf{Y}\Vert_F^2 \\
\text{subject to} &\quad \mathbf{A} \geq 0
\end{align}
$$

This is a **quadratic programming problem** with inequality constraints. The analytical solution is more complex, but we can use iterative methods.


_______

#### What is a Quadratic Programming Problem?

A **Quadratic Programming Problem** is an optimization problem where:
- The **objective function** is quadratic in the decision variables
- The **constraints** are linear (equality and/or inequality constraints)

The general form of a QPP is:
$$
\begin{align}
\min\_{\mathbf{x}} &\quad \frac{1}{2}\mathbf{x}^T\mathbf{Q}\mathbf{x} + \mathbf{c}^T\mathbf{x} \\
\text{subject to} &\quad \mathbf{A}\_{eq}\mathbf{x} = \mathbf{b}\_{eq} \\
&\quad \mathbf{A}\_{ineq}\mathbf{x} \leq \mathbf{b}\_{ineq}
\end{align}
$$

In our case:
- $\mathbf{Q} = \mathbf{S}^T\mathbf{S}$ (positive semi-definite matrix)
- The constraint $\mathbf{A} \geq 0$ represents simple bounds
- QPPs are **convex** when $\mathbf{Q}$ is positive semi-definite, ensuring a unique global minimum

#### Projected Gradient Algorithm

Since analytical solutions for QPPs with inequality constraints can be complex, we use iterative methods. The **projected gradient algorithm** is particularly effective for problems with simple constraints like non-negativity.

The algorithm works as follows:

1. **Gradient Step**: Take a standard gradient descent step
   $$\tilde{\mathbf{A}}^{(k+1)} = \mathbf{A}^{(k)} - \alpha \nabla f(\mathbf{A}^{(k)})$$

2. **Projection Step**: Project the result onto the feasible set
   $$\mathbf{A}^{(k+1)} = \text{Proj}\_{\mathcal{C}}(\tilde{\mathbf{A}}^{(k+1)})$$

where $\mathcal{C}$ is the constraint set and $\text{Proj}\_{\mathcal{C}}(\cdot)$ is the orthogonal projection operator.

For our problem:
- The gradient is: $\nabla f(\mathbf{A}) = 2\mathbf{S}^T(\mathbf{S}\mathbf{A} - \mathbf{Y})$
- The projection onto the non-negative orthant is: $\text{Proj}\_{\mathbb{R}_+}(\mathbf{x})_i = \max(0, x_i)$

The projection step ensures that all iterates remain feasible while the gradient step minimizes the objective function.

**Tasks:**

1. **Derive the gradient** of the objective function $f(\mathbf{A}) = \Vert\mathbf{S}\mathbf{A} - \mathbf{Y}\Vert_F^2$ with respect to $\mathbf{A}$.

2. **Implement a projected gradient descent algorithm** to solve the non-negativity constrained problem.

3. **Apply the non-negativity constrained method** to the Indian Pines dataset. Be carefull on how you choose the descent step size. Compare results with previous methods.

### 3. Fully Constrained Least Squares

Finally, let's combine both constraints:

$$
\begin{align}
\min\_{\mathbf{A}} &\quad \Vert\mathbf{S}\mathbf{A} - \mathbf{Y}\Vert_F^2 \\
\text{subject to} &\quad \mathbf{1}^T\mathbf{A} = \mathbf{1}^T \\
&\quad \mathbf{A} \geq 0
\end{align}
$$

#### The Simplex and Simplex Projection

The set defined by the intersection of the sum-to-one constraint and the non-negativity constraint defines the **unit simplex**.

**Definition**: The unit simplex in $\mathbb{R}^K$ is defined as:
$$\Delta^{K-1} = \left\\{\mathbf{x} \in \mathbb{R}^K : \sum\_{i=1}^K x_i = 1,  \forall i \leq K \  x_i \geq 0 \right\\}$$

**Geometric Interpretation**:
- In 2D ($K=2$): The simplex is a line segment from $(1,0)$ to $(0,1)$
- In 3D ($K=3$): The simplex is a triangle with vertices at $(1,0,0)$, $(0,1,0)$, and $(0,0,1)$
- In general: The simplex is a $(K-1)$-dimensional polytope

**Physical Meaning**: In our context, each column of $\mathbf{A}$ (representing abundances for one pixel) must lie on the simplex, ensuring that abundances are non-negative and sum to one.

#### Projection onto the Simplex

The **projection of a point $\mathbf{v}$ onto the simplex** is the closest point in the simplex to $\mathbf{v}$ in the Euclidean sense:
$$\text{Proj}\_{\Delta^{K-1}}(\mathbf{v}) = \arg\min\_{\mathbf{x} \in \Delta^{K-1}} \Vert\mathbf{x} - \mathbf{v}\Vert_2^2$$

{{<hiddenhint "An algorithm for simplex projection (Duchi et al., 2008)">}}

1. **Sort** the coordinates: $v_1 \geq v_2 \geq \ldots \geq v_K$
2. **Find the threshold**: $\rho = \max\{j : v_j - \frac{1}{j}(\sum\_{i=1}^j v_i - 1) > 0\}$
3. **Compute the Lagrange multiplier**: $\lambda = \frac{1}{\rho}(\sum\_{i=1}^\rho v_i - 1)$
4. **Project**: $[\text{Proj}\_{\Delta^{K-1}}(\mathbf{v})]_i = \max(0, v_i - \lambda)$
{{</hiddenhint>}}
**Tasks:**

1. **Implement a function that perform projection on the simplex**:

2. **Modify the previous projected gradient descent algorithm to the Fully Constrained Least Square problem.**

3. **Apply the fully constrained method** to the Indian Pines dataset.

4. **Compare all methods** (unconstrained, sum-to-one, non-negativity, fully constrained) in terms of:
   - Reconstruction quality (SAM, RMSE, SSIM)
   - Physical meaningfulness of abundances
   - Computational efficiency
   - Visual quality of abundance maps

5. **Analyze the convergence behavior** of the different projected gradient algorithms. Plot the objective function value vs. iteration number.

## IV - Blind Hyperspectral Unmixing (Bonus)

*This section is intended for advanced students who have successfully completed the previous sections.*

In the previous sections, we assumed that the endmembers were known. In practice, this is often not the case, and we need to estimate both the endmembers and abundances simultaneously. This is called **blind hyperspectral unmixing** or **blind source separation**.

The optimization problem becomes:

$$
\begin{align}
\min\_{\mathbf{A}, \mathbf{S}} &\quad \Vert\mathbf{S}\mathbf{A} - \mathbf{Y}\Vert\_F^2 \\
\text{subject to} &\quad \mathbf{1}^T\mathbf{A} = \mathbf{1}^T \\
&\quad \mathbf{A} \geq 0 \\
&\quad \mathbf{S} \geq 0
\end{align}
$$

This problem is significantly more challenging because:
- It is **non-convex** in the joint variables $(\mathbf{A}, \mathbf{S})$
- Multiple local minima exist
- The solution is not unique (scaling ambiguity)

### Block Coordinate Descent Algorithm

Since the problem is not jointly convex, but is convex in each variable when the other is fixed, we can use **Block Coordinate Descent (BCD)**. This approach alternates between optimizing blocks of variables while keeping others fixed.

**Algorithm Structure:**
1. **Initialize** $\mathbf{S}^{(0)}$ and $\mathbf{A}^{(0)}$
2. **For** $k = 0, 1, 2, \ldots$ **until convergence:**
   - **Fix** $\mathbf{S} = \mathbf{S}^{(k)}$ and solve for $\mathbf{A}^{(k+1)}$:
     $$\mathbf{A}^{(k+1)} = \arg\min\_{\mathbf{A}} \Vert\mathbf{S}^{(k)}\mathbf{A} - \mathbf{Y}\Vert_F^2 \quad \text{s.t. } \mathbf{1}^T\mathbf{A} = \mathbf{1}^T, \mathbf{A} \geq 0$$
   
   - **Fix** $\mathbf{A} = \mathbf{A}^{(k+1)}$ and solve for $\mathbf{S}^{(k+1)}$:
     $$\mathbf{S}^{(k+1)} = \arg\min\_{\mathbf{S}} \Vert\mathbf{S}\mathbf{A}^{(k+1)} - \mathbf{Y}\Vert_F^2 \quad \text{s.t. } \mathbf{S} \geq 0$$

**Key Insight:** Each subproblem is a constrained least squares problem that we know how to solve from Section III!

**Tasks:**

1. **Analyze the subproblems:**
   - What type of optimization problem is the $\mathbf{A}$-subproblem? Which method from Section III can you use?
   - What type of optimization problem is the $\mathbf{S}$-subproblem? How does it differ from the abundance estimation?

2. **Implement the Block Coordinate Descent algorithm:**

3. **Initialization strategies:** Implement at least two initialization methods:
   - **Random initialization:** Random positive values with proper normalization
   - **Supervised initialization:** Use the endmembers provided on section III.

4. **Convergence analysis:** 
   - Implement a convergence criterion based on the relative change in objective function
   - Plot the objective function value vs. iteration number
   - Compare convergence behavior with different initializations

6. **Algorithm analysis:**
   - What are the main limitations of this approach?
   - How sensitive is the algorithm to initialization?
   - What happens when the number of endmembers $K$ is incorrectly specified?

### Limitations and Advanced Methods

The Block Coordinate Descent approach presented here provides a good introduction to blind unmixing, but it has several limitations:

- **Local minima:** The algorithm may converge to poor local solutions
- **Initialization dependence:** Results can vary significantly with different initializations  
- **Scaling ambiguity:** Solutions are not unique up to scaling factors
- **Slow convergence:** Simple alternating optimization can be slow

**Advanced methods exist** that address these limitations and provide better performance:

- **Non-negative Matrix Factorization (NMF)** with multiplicative updates
- **Independent Component Analysis (ICA)** for statistical independence
...

These advanced methods often incorporate additional prior knowledge, regularization terms, or sophisticated optimization techniques to achieve more robust and accurate unmixing results.