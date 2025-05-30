---
title: Differentiation
weight: 2
chapter: 0
math: true
---

# Differentiation in Multiple Dimensions

## 1 - Introduction

> Differentiation provides the mathematical framework for understanding how functions change locally. While single-variable calculus introduces derivatives, most applications require working with functions of multiple variables. This chapter extends differentiation concepts to multivariate and matrix-valued functions, building the tools needed for optimization and analysis in higher dimensions.

## 2 - Monovariate Reminders

### Derivative of a Function

{{<definition "Derivative" derivative_definition>}}
The **derivative** of a function $f:\mathbb{R}\to\mathbb{R}$ at a point $x_0$ is defined as:
$$f'(x_0) = \lim\_{h\to 0} \frac{f(x_0 + h) - f(x_0)}{h}$$
provided this limit exists.
{{</definition>}}

The derivative represents the instantaneous rate of change of the function at a specific point. Geometrically, $f'(x_0)$ gives the slope of the tangent line to the curve $y = f(x)$ at the point $(x_0, f(x_0))$. This tangent line provides the best linear approximation to the function near $x_0$.

For practical computation, we use two fundamental rules:

- **Product rule**: $(uv)' = u'v + uv'$
- **Chain rule**: $(f(g(x)))' = f'(g(x))g'(x)$

These rules allow us to differentiate complex expressions by breaking them down into simpler components.

## 3 - Extension to Multivariate Setup: $f:\mathbb{R}^d \to \mathbb{R}$

### Limits and Continuity

{{<definition "Open Disk" open_disk>}}
An **open disk** of radius $\epsilon > 0$ centered at a point $\mathbf{x}_0 \in \mathbb{R}^d$ is defined as:
$$\mathcal{B}(\mathbf{x}_0, \epsilon) = \{\mathbf{x} \in \mathbb{R}^d : \|\mathbf{x} - \mathbf{x}_0\|_2 < \epsilon\}$$
{{</definition>}}

{{<definition "Limit" limit_definition>}}
The **limit** of a function $f:\mathbb{R}^d \to \mathbb{R}$ at a point $\mathbf{x}_0$ is defined as:
$$\lim\_{\mathbf{x} \to \mathbf{x}_0} f(\mathbf{x}) = L$$
if for every $\epsilon > 0$, there exists $\delta > 0$ such that whenever $\|\mathbf{x} - \mathbf{x}_0\|_2 < \delta$, we have $|f(\mathbf{x}) - L| < \epsilon$.
{{</definition>}}

A function is **continuous** at a point $\mathbf{x}_0$ if $\lim\_{\mathbf{x} \to \mathbf{x}_0} f(\mathbf{x}) = f(\mathbf{x}_0)$. These definitions generalize the single-variable concepts using the Euclidean norm to measure distances in $\mathbb{R}^d$.

### Directional Derivative

{{<definition "Directional Derivative" directional_derivative>}}
The **directional derivative** of a function $f:\mathbb{R}^d \to \mathbb{R}$ at a point $\mathbf{x}_0$ in the direction of a vector $\mathbf{v} \in \mathbb{R}^d$ is defined as:
$$Df(\mathbf{x}_0)[\mathbf{v}] = \lim\_{h \to 0} \frac{f(\mathbf{x}_0 + h\mathbf{v}) - f(\mathbf{x}_0)}{h}$$
{{</definition>}}

When $\|\mathbf{v}\|_2 = 1$, the directional derivative $Df(\mathbf{x}_0)[\mathbf{v}]$ represents the rate of change of $f$ in the direction of $\mathbf{v}$ at the point $\mathbf{x}_0$. This generalizes the concept of derivative to any direction in the input space.

We also use the notation $\nabla\_{\mathbf{v}}f(\mathbf{x}_0)$ for the directional derivative.

### Gradient

{{<definition "Gradient" gradient_definition>}}
The **gradient** of a function $f:\mathbb{R}^d \to \mathbb{R}$ at a point $\mathbf{x}_0$ is defined as the vector of all directional derivatives in the standard basis directions:
$$\nabla f(\mathbf{x}_0) = \left( Df(\mathbf{x}_0)[\mathbf{e}_1], Df(\mathbf{x}_0)[\mathbf{e}_2], \ldots, Df(\mathbf{x}_0)[\mathbf{e}_d] \right)^\mathrm{T}$$
where $\{\mathbf{e}_1, \ldots, \mathbf{e}_d\}$ is the standard basis of $\mathbb{R}^d$.
{{</definition>}}

The gradient points in the direction of steepest ascent of the function $f$ at the point $\mathbf{x}_0$. It encodes all the first-order information about how the function changes locally.

For any vector $\mathbf{v} \in \mathbb{R}^d$, the directional derivative can be expressed as:
$$Df(\mathbf{x}_0)[\mathbf{v}] = \nabla f(\mathbf{x}_0)^\mathrm{T} \mathbf{v}$$

This shows that the gradient contains all the information needed to compute directional derivatives in any direction.

### Gradient and Partial Derivatives

{{<definition "Partial Derivative" partial_derivative>}}
The **partial derivative** of a function $f:\mathbb{R}^d \to \mathbb{R}$ with respect to the $i$-th variable is defined as:
$$\frac{\partial f}{\partial x_i}(\mathbf{x}_0) = \lim\_{h \to 0} \frac{f(\mathbf{x}_0 + h\mathbf{e}_i) - f(\mathbf{x}_0)}{h}$$
where $\mathbf{e}_i$ is the $i$-th standard basis vector.
{{</definition>}}

The gradient can be expressed in terms of partial derivatives as:
$$\nabla f(\mathbf{x}_0) = \left( \frac{\partial f}{\partial x_1}(\mathbf{x}_0), \frac{\partial f}{\partial x_2}(\mathbf{x}_0), \ldots, \frac{\partial f}{\partial x_d}(\mathbf{x}_0) \right)^\mathrm{T}$$

This representation makes it clear that the gradient is a vector containing all the partial derivatives of the function at the point $\mathbf{x}_0$.

### Gradient Properties and Practical Computation

When computing gradients in practice, we use the following rules:

{{<theorem "Product Rule for Gradients">}}
Let $g:\mathbb{R}^d \to \mathbb{R}$ and $h:\mathbb{R}^d \to \mathbb{R}$ be two functions. Then the gradient of their product $f(\mathbf{x}) = g(\mathbf{x})h(\mathbf{x})$ is:
$$\nabla f(\mathbf{x}) = g(\mathbf{x})\nabla h(\mathbf{x}) + h(\mathbf{x})\nabla g(\mathbf{x})$$
{{</theorem>}}

{{<theorem "Chain Rule for Gradients">}}
For composition of functions, we have two main cases:
1. If $f=h\circ g$ where $h:\mathbb{R}\to\mathbb{R}$ and $g:\mathbb{R}^d\to\mathbb{R}$, then:
   $$\nabla f(\mathbf{x}) = h'(g(\mathbf{x}))\nabla g(\mathbf{x})$$
   where $h'$ is the derivative of $h$.
2. If $f=h\circ g$ where $h:\mathbb{R}^d\to\mathbb{R}$ and $g:\mathbb{R}^{d'}\to\mathbb{R}^d$, we need the more general chain rule discussed later.
{{</theorem>}}

### Hessian Matrix

{{<definition "Hessian Matrix" hessian_definition>}}
The **Hessian matrix** of a function $f:\mathbb{R}^d \to \mathbb{R}$ at a point $\mathbf{x}_0$ is defined as the square matrix of second-order partial derivatives:
$$\mathbf{H}(\mathbf{x}_0) = \begin{pmatrix}
\frac{\partial^2 f}{\partial x_1^2}(\mathbf{x}_0) & \frac{\partial^2 f}{\partial x_1 \partial x_2}(\mathbf{x}_0) & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_d}(\mathbf{x}_0) \\\\
\frac{\partial^2 f}{\partial x_2 \partial x_1}(\mathbf{x}_0) & \frac{\partial^2 f}{\partial x_2^2}(\mathbf{x}_0) & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_d}(\mathbf{x}_0) \\\\
\vdots & \vdots & \ddots & \vdots \\\\
\frac{\partial^2 f}{\partial x_d \partial x_1}(\mathbf{x}_0) & \frac{\partial^2 f}{\partial x_d \partial x_2}(\mathbf{x}_0) & \cdots & \frac{\partial^2 f}{\partial x_d^2}(\mathbf{x}_0)
\end{pmatrix}$$
{{</definition>}}

The Hessian matrix captures the second-order behavior of the function, providing information about its curvature at the point $\mathbf{x}_0$.

**Exercise 1**: Compute the gradient and Hessian matrix of the function $f(x,y) = x^2 + 3xy + y^2$ at the point $(1,2)$.

**Exercise 2**: Using the chain rule, compute the gradient of $f(\mathbf{x}) = \left(\sum\_{i=1}^{d}x_i^2\right)^{1/2}$.

### Hessian Matrix Properties

The Hessian matrix has several important properties:

- **Symmetry**: If $f$ is twice continuously differentiable, then $\mathbf{H}(\mathbf{x}_0) = \mathbf{H}(\mathbf{x}_0)^\mathrm{T}$ because mixed partial derivatives are equal: $\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}$.

- **Curvature information**: The eigenvalues of the Hessian determine the local curvature:
  - All eigenvalues positive: $f$ is locally convex at $\mathbf{x}_0$
  - All eigenvalues negative: $f$ is locally concave at $\mathbf{x}_0$
  - Mixed positive and negative eigenvalues: $f$ has a saddle point at $\mathbf{x}_0$

**Exercise 3 (Rosenbrock function)**: The Rosenbrock function is defined as:
$$f(x,y) = (a - x)^2 + b(y - x^2)^2$$
where $a$ and $b$ are constants (commonly $a=1$ and $b=100$).

1. Compute the gradient $\nabla f(x,y)$ and find stationary points.
2. Compute the Hessian matrix $\mathbf{H}(x,y)$ and analyze local curvature at the stationary points.

## 4 - Multivariate Case: $f:\mathbb{R}^d \to \mathbb{R}^p$

### Multivariate Functions

{{<definition "Vector-Valued Function" vector_valued_function>}}
A **vector-valued function** $f:\mathbb{R}^d \to \mathbb{R}^p$ maps a vector $\mathbf{x} \in \mathbb{R}^d$ to a vector $\mathbf{y} \in \mathbb{R}^p$. We can write:
$$f(\mathbf{x}) = \begin{pmatrix}
f_1(\mathbf{x}) \\\\
f_2(\mathbf{x}) \\\\
\vdots \\\\
f_p(\mathbf{x})
\end{pmatrix}$$
where each component $f_i:\mathbb{R}^d \to \mathbb{R}$ is a scalar function.
{{</definition>}}

### Gradient and Jacobian

For scalar-valued functions, we defined the gradient. For vector-valued functions, we need the Jacobian matrix.

{{<definition "Jacobian Matrix" jacobian_definition>}}
The **Jacobian matrix** of a function $f:\mathbb{R}^d \to \mathbb{R}^p$ is defined as:
$$\mathbf{J}_f(\mathbf{x}) = \begin{pmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_d} \\\\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_d} \\\\
\vdots & \vdots & \ddots & \vdots \\\\
\frac{\partial f_p}{\partial x_1} & \frac{\partial f_p}{\partial x_2} & \cdots & \frac{\partial f_p}{\partial x_d}
\end{pmatrix} \in \mathbb{R}^{p \times d}$$
{{</definition>}}

The Jacobian matrix generalizes the gradient to vector-valued functions. Each row is the gradient of one component function.

### Jacobian and Directional Derivative

The directional derivative of a vector-valued function $f:\mathbb{R}^d \to \mathbb{R}^p$ in the direction of a vector $\mathbf{v} \in \mathbb{R}^d$ is:
$$Df(\mathbf{x})[\mathbf{v}] = \mathbf{J}_f(\mathbf{x})\mathbf{v} = \begin{pmatrix} \nabla f_1(\mathbf{x})^T \mathbf{v} \\\\ \nabla f_2(\mathbf{x})^T \mathbf{v} \\\\ \vdots \\\\ \nabla f_p(\mathbf{x})^T \mathbf{v} \end{pmatrix} \in \mathbb{R}^p$$

This shows how the Jacobian matrix encodes all directional derivative information.

### Chain Rule for Composition of Functions

{{<theorem "General Chain Rule">}}
If $f:\mathbb{R}^d \to \mathbb{R}^p$ and $g:\mathbb{R}^m \to \mathbb{R}^d$, then the composition $h = f \circ g : \mathbb{R}^m \to \mathbb{R}^p$ is defined as:
$$h(\mathbf{y}) = f(g(\mathbf{y}))$$
The Jacobian of $h$ can be computed using the chain rule:
$$\mathbf{J}_h(\mathbf{y}) = \mathbf{J}_f(g(\mathbf{y})) \mathbf{J}_g(\mathbf{y})$$
where $\mathbf{J}_h(\mathbf{y}) \in \mathbb{R}^{p \times m}$, $\mathbf{J}_f(g(\mathbf{y})) \in \mathbb{R}^{p \times d}$, and $\mathbf{J}_g(\mathbf{y}) \in \mathbb{R}^{d \times m}$.
{{</theorem>}}

### Chain Rule: Special Cases

**Case 1**: If $f:\mathbb{R}^d \to \mathbb{R}$ and $g:\mathbb{R}^m \to \mathbb{R}^d$, then for $h = f \circ g : \mathbb{R}^m \to \mathbb{R}$:
$$\nabla h(\mathbf{y}) = \mathbf{J}_g(\mathbf{y})^T \nabla f(g(\mathbf{y}))$$

**Case 2**: If $f:\mathbb{R} \to \mathbb{R}$ and $g:\mathbb{R}^m \to \mathbb{R}$, then for $h = f \circ g : \mathbb{R}^m \to \mathbb{R}$:
$$\nabla h(\mathbf{y}) = f'(g(\mathbf{y})) \nabla g(\mathbf{y})$$

### Worked Examples

**Example 1**: Given:
- $f(\mathbf{x}) = \mathbf{x}^T\mathbf{x}$ where $f: \mathbb{R}^2 \to \mathbb{R}$
- $g(\mathbf{y}) = \begin{pmatrix} y_1 + y_2 \\\\ y_1 - y_2 \end{pmatrix}$ where $g: \mathbb{R}^2 \to \mathbb{R}^2$
- $h = f \circ g$

Find $\nabla h(\mathbf{y})$ using the chain rule.

**Solution**:
1. First, $\nabla f(\mathbf{x}) = 2\mathbf{x}$
2. The Jacobian is $\mathbf{J}_g(\mathbf{y}) = \begin{pmatrix} 1 & 1 \\\\ 1 & -1 \end{pmatrix}$
3. Applying the chain rule:
   $$\nabla h(\mathbf{y}) = \mathbf{J}_g(\mathbf{y})^T \nabla f(g(\mathbf{y})) = \begin{pmatrix} 1 & 1 \\\\ 1 & -1 \end{pmatrix} \cdot 2g(\mathbf{y})$$
   $$= 2\begin{pmatrix} 1 & 1 \\\\ 1 & -1 \end{pmatrix}\begin{pmatrix} y_1 + y_2 \\ y_1 - y_2 \end{pmatrix} = \begin{pmatrix} 4y_1 \\ 4y_2 \end{pmatrix}$$

**Example 2 (General Quadratic Forms)**: Given:
- $f(\mathbf{x}) = \mathbf{x}^T\mathbf{A}\mathbf{x} + \mathbf{b}^T\mathbf{x}$ where $\mathbf{A}$ is symmetric
- $g(\mathbf{y}) = \mathbf{C}\mathbf{y}$ (linear transformation)

Find $\nabla h(\mathbf{y})$ for $h = f \circ g$.

**Solution**:
1. $\nabla f(\mathbf{x}) = 2\mathbf{A}\mathbf{x} + \mathbf{b}$
2. $\mathbf{J}_g(\mathbf{y}) = \mathbf{C}$
3. Therefore:
   $$\nabla h(\mathbf{y}) = \mathbf{C}^T [2\mathbf{A}(\mathbf{C}\mathbf{y}) + \mathbf{b}] = 2\mathbf{C}^T\mathbf{A}\mathbf{C}\mathbf{y} + \mathbf{C}^T\mathbf{b}$$

## 5 - Matrix Functions: $f:\mathbb{R}^{m \times n} \to \mathbb{R}$

### Fréchet Derivative

{{<definition "Fréchet Differentiability" frechet_differentiability>}}
A function $f:\mathbb{R}^{m \times n}\to\mathbb{R}^{p \times q}$ is **Fréchet differentiable** at $\mathbf{X}$ if there exists a linear mapping $Df(\mathbf{X}):\mathbb{R}^{m \times n}\to\mathbb{R}^{p \times q}$ such that
$$\lim\_{\|\mathbf{V}\|_F\to 0} \frac{\|f(\mathbf{X}+\mathbf{V}) - f(\mathbf{X}) - Df(\mathbf{X})[\mathbf{V}]\|_F}{\|\mathbf{V}\|_F} = 0$$
{{</definition>}}

The Fréchet derivative can also be characterized using the **Gateaux derivative**:
$$Df(\mathbf{X})[\mathbf{V}] = \left.\frac{d}{dt}\right|\_{t=0} f(\mathbf{X}+t\mathbf{V}) = \lim\_{t\to 0} \frac{f(\mathbf{X}+t\mathbf{V}) - f(\mathbf{X})}{t}$$

If this limit is not linear in $\mathbf{V}$, then $f$ is not Fréchet differentiable.

Often it is useful to see this derivative as a linear operator such that:
$$ \mathbf{D} f(\mathbf{X})[\boldsymbol{\xi}] = f(\mathbf{X}+\mathbf{\xi}) - f(\mathbf{X}) + o(\lVert\boldsymbol{\xi}\rVert)$$


### Matrix-to-Scalar Functions

For a function $f:\mathbb{R}^{m \times n} \to \mathbb{R}$, the directional derivative at $\mathbf{X}$ in direction $\mathbf{V}$ is:
$$Df(\mathbf{X})[\mathbf{V}] = \lim\_{h \to 0} \frac{f(\mathbf{X} + h\mathbf{V}) - f(\mathbf{X})}{h}$$

{{<definition "Matrix Gradient" matrix_gradient>}}
For $f:\mathbb{R}^{m \times n} \to \mathbb{R}$, the **gradient** $\nabla f(\mathbf{X}) \in \mathbb{R}^{m \times n}$ satisfies:
$$Df(\mathbf{X})[\mathbf{V}] = \mathrm{Tr}(\nabla f(\mathbf{X})^\mathrm{T} \mathbf{V})$$
where $\mathrm{Tr}(\cdot)$ denotes the trace of a matrix.
{{</definition>}}

The gradient can be computed element-wise as:
$$\nabla f(\mathbf{X}) = \begin{pmatrix}
\frac{\partial f}{\partial X\_{11}} & \frac{\partial f}{\partial X\_{12}} & \cdots & \frac{\partial f}{\partial X\_{1n}} \\\\
\frac{\partial f}{\partial X\_{21}} & \frac{\partial f}{\partial X\_{22}} & \cdots & \frac{\partial f}{\partial X\_{2n}} \\\\
\vdots & \vdots & \ddots & \vdots \\\\
\frac{\partial f}{\partial X\_{m1}} & \frac{\partial f}{\partial X\_{m2}} & \cdots & \frac{\partial f}{\partial X\_{mn}}
\end{pmatrix}$$

### Examples of Matrix-to-Scalar Functions

**Example 1**: $f(\mathbf{X}) = \|\mathbf{X}\|_F^2 = \mathrm{Tr}(\mathbf{X}^\mathrm{T}\mathbf{X})$

Using the Gateaux derivative:
$$Df(\mathbf{X})[\mathbf{V}] = \left.\frac{d}{dt}\right|\_{t=0} \mathrm{Tr}((\mathbf{X}+t\mathbf{V})^\mathrm{T}(\mathbf{X}+t\mathbf{V}))$$

Expanding and differentiating:
$$= \left.\frac{d}{dt}\right|\_{t=0} [\mathrm{Tr}(\mathbf{X}^\mathrm{T}\mathbf{X}) + 2t\mathrm{Tr}(\mathbf{X}^\mathrm{T}\mathbf{V}) + t^2\mathrm{Tr}(\mathbf{V}^\mathrm{T}\mathbf{V})]$$
$$= 2\mathrm{Tr}(\mathbf{X}^\mathrm{T}\mathbf{V})$$

Therefore: $\nabla f(\mathbf{X}) = 2\mathbf{X}$

**Example 2**: $f(\mathbf{X}) = \log\det(\mathbf{X})$ (for invertible $\mathbf{X}$)

For this function:
$$Df(\mathbf{X})[\mathbf{V}] = \left.\frac{d}{dt}\right|\_{t=0} \log\det(\mathbf{X}+t\mathbf{V}) = \mathrm{Tr}(\mathbf{X}^{-1}\mathbf{V})$$

Therefore: $\nabla f(\mathbf{X}) = \mathbf{X}^{-\mathrm{T}}$

## 6 - Matrix Functions: $f:\mathbb{R}^{m \times n} \to \mathbb{R}^{p \times q}$

### Matrix-to-Matrix Functions

For a function $f:\mathbb{R}^{m \times n} \to \mathbb{R}^{p \times q}$, the directional derivative $Df(\mathbf{X})[\mathbf{V}]$ is a linear mapping from $\mathbb{R}^{m \times n}$ to $\mathbb{R}^{p \times q}$.

Since $Df(\mathbf{X})$ is linear, there exists a matrix $\mathbf{M}\_{\mathbf{X}} \in \mathbb{R}^{pq \times mn}$ such that:
$$\mathrm{vec}(Df(\mathbf{X})[\mathbf{V}]) = \mathbf{M}\_{\mathbf{X}} \mathrm{vec}(\mathbf{V})$$
where $\mathrm{vec}(\cdot)$ stacks matrix columns into a vector.

This representation transforms the problem of computing matrix derivatives into standard matrix-vector multiplication. The matrix $\mathbf{M}_{\mathbf{X}}$ is sometimes called the **derivative matrix** or **Jacobian matrix** of the vectorized function.

The power of this representation becomes clear when combined with the Kronecker product identity:

{{<theorem "Kronecker Product Identity">}}
For matrices $\mathbf{A} \in \mathbb{R}^{p \times m}$, $\mathbf{B} \in \mathbb{R}^{n \times q}$, and $\mathbf{X} \in \mathbb{R}^{m \times n}$:
$$\mathrm{vec}(\mathbf{A}\mathbf{X}\mathbf{B}) = (\mathbf{B}^\mathrm{T} \otimes \mathbf{A})  \mathrm{vec}(\mathbf{X})$$
{{</theorem>}}

**Example**: Consider $f(\mathbf{X}) = \mathbf{A}\mathbf{X}\mathbf{B}$ where $\mathbf{A} \in \mathbb{R}^{p \times m}$ and $\mathbf{B} \in \mathbb{R}^{n \times q}$ are fixed matrices.

To find the derivative, we compute:
$$Df(\mathbf{X})[\mathbf{V}] = f(\mathbf{X} + \mathbf{V}) - f(\mathbf{X}) = \mathbf{A}(\mathbf{X} + \mathbf{V})\mathbf{B} - \mathbf{A}\mathbf{X}\mathbf{B} = \mathbf{A}\mathbf{V}\mathbf{B}$$

Using the Kronecker product identity:
$$\mathrm{vec}(Df(\mathbf{X})[\mathbf{V}]) = \mathrm{vec}(\mathbf{A}\mathbf{V}\mathbf{B}) = (\mathbf{B}^\mathrm{T} \otimes \mathbf{A}) \mathrm{vec}(\mathbf{V})$$

Therefore, $\mathbf{M}_{\mathbf{X}} = \mathbf{B}^\mathrm{T} \otimes \mathbf{A}$, which is independent of $\mathbf{X}$ since $f$ is linear.

### Vectorization Identities

Key identities for working with matrix derivatives:
- $\mathrm{vec}(\mathbf{A}\mathbf{B}\mathbf{C}) = (\mathbf{C}^\mathrm{T} \otimes \mathbf{A}) \mathrm{vec}(\mathbf{B})$
- $\mathrm{Tr}(\mathbf{A}\mathbf{B}) = \mathrm{vec}(\mathbf{A})^\mathrm{T}\mathrm{vec}(\mathbf{B})$
- $\mathrm{Tr}(\mathbf{A}^\mathrm{T}\mathbf{B}) = \mathrm{vec}(\mathbf{A})^\mathrm{T}\mathrm{vec}(\mathbf{B})$

where $\otimes$ denotes the Kronecker product.

### Examples of Matrix-to-Matrix Functions

**Example 1**: $f(\mathbf{X}) = \mathbf{X}^2$

Using the Gateaux derivative:
$$Df(\mathbf{X})[\mathbf{V}] = \left.\frac{d}{dt}\right|\_{t=0} (\mathbf{X}+t\mathbf{V})^2 = \mathbf{X}\mathbf{V} + \mathbf{V}\mathbf{X}$$

**Example 2**: $f(\mathbf{X}) = \mathbf{X}^{-1}$ (for invertible $\mathbf{X}$)

From the identity $\mathbf{X}\mathbf{X}^{-1} = \mathbf{I}$ and differentiating:
$$Df(\mathbf{X})[\mathbf{V}] = -\mathbf{X}^{-1}\mathbf{V}\mathbf{X}^{-1}$$

### Properties of Matrix Function Derivatives

The derivatives of matrix functions follow familiar rules:

- **Linearity**: For $f = \alpha g + \beta h$:
  $$Df(\mathbf{X})[\mathbf{V}] = \alpha \, Dg(\mathbf{X})[\mathbf{V}] + \beta \, Dh(\mathbf{X})[\mathbf{V}]$$

- **Product rule**: For $f(\mathbf{X}) = g(\mathbf{X}) \cdot h(\mathbf{X})$:
  $$Df(\mathbf{X})[\mathbf{V}] = Dg(\mathbf{X})[\mathbf{V}] \cdot h(\mathbf{X}) + g(\mathbf{X}) \cdot Dh(\mathbf{X})[\mathbf{V}]$$

- **Chain rule**: For $f(\mathbf{X}) = g(h(\mathbf{X}))$:
  $$Df(\mathbf{X})[\mathbf{V}] = Dg(h(\mathbf{X}))[Dh(\mathbf{X})[\mathbf{V}]]$$

