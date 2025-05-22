---
title: Introduction
weight: 2
---

# Introduction

## Notations

  Let us start by defining the notation used troughout all the lectures and practical labs.

### Basic Notation

Scalars are represented by italic letters (e.g., $x$, $y$, $\lambda$). Vectors are denoted by bold lowercase letters (e.g., $\mathbf{v}$, $\mathbf{x}$), while matrices are represented by bold uppercase letters (e.g., $\mathbf{A}$, $\mathbf{B}$). The dimensionality of a vector $\mathbf{v} \in \mathbb{R}^n$ indicates it contains $n$ elements, and similarly, a matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ has $m$ rows and $n$ columns.

### Matrix Operations

The transpose of a matrix $\mathbf{A}$ is denoted as $\mathbf{A}^\mathrm{T}$, which reflects the matrix across its diagonal. The trace of a square matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$, written as $\mathrm{tr}(\mathbf{A})$, is the sum of its diagonal elements, i.e., $\mathrm{tr}(\mathbf{A}) = \sum_{i=1}^{n} a_{ii}$. The determinant of $\mathbf{A}$ is represented as $\det(\mathbf{A})$ or $|\mathbf{A}|$. A matrix $\mathbf{A}$ is invertible if and only if $\det(\mathbf{A}) \neq 0$, and its inverse is denoted as $\mathbf{A}^{-1}$, satisfying $\mathbf{A}\mathbf{A}^{-1} = \mathbf{A}^{-1}\mathbf{A} = \mathbf{I}$, where $\mathbf{I}$ is the identity matrix.

### Vector Operations


The dot product between two vectors $\mathbf{a}$ and $\mathbf{b}$ of the same dimension is written as $\mathbf{a} \cdot \mathbf{b}$ or $\mathbf{a}^\mathrm{T}\mathbf{b}$, resulting in a scalar value.  

The p-norm of a vector $\mathbf{v}$ is denoted as $\lVert\mathbf{v}\rVert_p$ and defined as $\lVert\mathbf{v}\rVert_p = \left(\sum_{i=1}^{n} |v_i|^p\right)^{1/p}$ for $p \geq 1$, with common choices being $p=1$ (Manhattan norm), $p=2$ (Euclidean norm), and $p=\infty$ (maximum norm, defined as $\lVert\mathbf{v}\rVert_{\infty} = \max_i |v_i|$); when the subscript $p$ is omitted, as in $\lVert\mathbf{v}\rVert$, it is conventionally understood to refer to the Euclidean (L2) norm. The Euclidean norm (or length) of a vector $\mathbf{v}$ is represented as $\lVert\mathbf{v}\rVert$ or $\lVert\mathbf{v}\rVert_2$, defined as $\lVert\mathbf{v}\rVert = \sqrt{\mathbf{v}^\mathrm{T}\mathbf{v}} = \sqrt{\sum_{i=1}^{n} v_i^2}$. A unit vector in the direction of $\mathbf{v}$ is given by $\hat{\mathbf{v}} = \mathbf{v}/\lVert\mathbf{v}\rVert$, having a norm of 1.


### Eigenvalues and Eigenvectors

For a square matrix $\mathbf{A}$, a scalar $\lambda$ is an eigenvalue if there exists a non-zero vector $\mathbf{v}$ such that $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$. The vector $\mathbf{v}$ is called an eigenvector corresponding to the eigenvalue $\lambda$. The characteristic polynomial of $\mathbf{A}$ is defined as $p(\lambda) = \det(\lambda\mathbf{I} - \mathbf{A})$, and its roots are the eigenvalues of $\mathbf{A}$. The spectrum of $\mathbf{A}$, denoted by $\sigma(\mathbf{A})$, is the set of all eigenvalues of $\mathbf{A}$.

### Matrix Decompositions

The singular value decomposition (SVD) of a matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ is expressed as $\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\mathrm{T}$, where $\mathbf{U} \in \mathbb{R}^{m \times m}$ and $\mathbf{V} \in \mathbb{R}^{n \times n}$ are orthogonal matrices, and $\mathbf{\Sigma} \in \mathbb{R}^{m \times n}$ is a diagonal matrix containing the singular values of $\mathbf{A}$. The eigendecomposition of a diagonalizable matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ is given by $\mathbf{A} = \mathbf{P}\mathbf{\Lambda}\mathbf{P}^{-1}$, where $\mathbf{P}$ is a matrix whose columns are the eigenvectors of $\mathbf{A}$, and $\mathbf{\Lambda}$ is a diagonal matrix containing the corresponding eigenvalues.

### Multivariate Calculus

The gradient of a scalar function $f: \mathbb{R}^n \rightarrow \mathbb{R}$ is denoted as $\nabla f$ or $\mathrm{grad}(f)$, resulting in a vector of partial derivatives $\nabla f = [\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}]^\mathrm{T}$. The Jacobian matrix of a vector-valued function $\mathbf{f}: \mathbb{R}^n \rightarrow \mathbb{R}^m$ is represented as $\mathbf{J}_\mathbf{f}$ or $\nabla \mathbf{f}^\mathrm{T}$, where $ (\mathbf{J}\_\mathbf{f})\_{ij} = \frac{\partial f\_i}{\partial x\_j} $.

The Hessian matrix of a twice-differentiable scalar function $f: \mathbb{R}^n \rightarrow \mathbb{R}$ is denoted as $\mathbf{H}\_f$ or $\nabla^2 f$, where $(\mathbf{H}\_f)\_{ij} = \frac{\partial^2 f}{\partial x\_i \partial x\_j}$.


### Special Matrices and Properties

A symmetric matrix satisfies $\mathbf{A} = \mathbf{A}^\mathrm{T}$, while a skew-symmetric matrix has $\mathbf{A} = -\mathbf{A}^\mathrm{T}$. An orthogonal matrix $\mathbf{Q}$ satisfies $\mathbf{Q}^\mathrm{T}\mathbf{Q} = \mathbf{Q}\mathbf{Q}^\mathrm{T} = \mathbf{I}$, meaning its inverse equals its transpose: $\mathbf{Q}^{-1} = \mathbf{Q}^\mathrm{T}$. A matrix $\mathbf{A}$ is positive definite if $\mathbf{x}^\mathrm{T}\mathbf{A}\mathbf{x} > 0$ for all non-zero vectors $\mathbf{x}$, and positive semidefinite if $\mathbf{x}^\mathrm{T}\mathbf{A}\mathbf{x} \geq 0$.

### Derivatives of Matrix Expressions

The derivative of a scalar function with respect to a vector $\mathbf{x}$ is denoted as $\frac{\partial f}{\partial \mathbf{x}}$, resulting in a vector of the same dimension as $\mathbf{x}$. For matrix functions, the derivative with respect to a matrix $\mathbf{X}$ is written as $\frac{\partial f}{\partial \mathbf{X}}$, producing a matrix of the same dimensions as $\mathbf{X}$. Common matrix derivatives include $\frac{\partial}{\partial \mathbf{X}}\mathrm{tr}(\mathbf{AX}) = \mathbf{A}^\mathrm{T}$ and $\frac{\partial}{\partial \mathbf{x}}(\mathbf{x}^\mathrm{T}\mathbf{A}\mathbf{x}) = \mathbf{A}\mathbf{x} + \mathbf{A}^\mathrm{T}\mathbf{x}$ (with $\mathbf{A}\mathbf{x} + \mathbf{A}^\mathrm{T}\mathbf{x} = 2\mathbf{A}\mathbf{x}$ when $\mathbf{A}$ is symmetric).


