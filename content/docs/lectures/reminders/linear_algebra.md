---
title: Linear Algebra
weight: 1
chapter: 0
math: true
---

# Fundamentals of Linear Algebra


## 1 - Introduction
> Linear algebra is one of the foundational branches of mathematics, with applications spanning from engineering and computer science to economics and physics. This document is extracted from the textbook **Matrix Differential Calculus with Applications in Statistics and Econometrics** from Jan R. Magnus; Heinz Neudecker, adapting the notations to the ones used in the lectures.



In this chapter, we summarize some of the well-known definitions and theorems of matrix algebra. Most of the theorems will be proved.

## 2 - Sets

{{<definition "Set" set_definition>}}
A **set** is a collection of objects, called the elements (or members) of the set. We write $x \in S$ to mean '$x$ is an element of $S$' or '$x$ belongs to $S$'. If $x$ does not belong to $S$, we write $x \notin S$. The set that contains no elements is called the **empty set**, denoted by $\emptyset$.
{{</definition>}}

Sometimes a set can be defined by displaying the elements in braces. For example, $A=\{0,1\}$ or

$$
\mathbb{N}=\{1,2,3, \ldots\}
$$

Notice that $A$ is a finite set (contains a finite number of elements), whereas $\mathbb{N}$ is an infinite set. If $P$ is a property that any element of $S$ has or does not have, then

$$
\{x: x \in S, x \text { satisfies } P\}
$$

denotes the set of all the elements of $S$ that have property $P$.

{{<definition "Subset" subset_definition>}}
A set $A$ is called a **subset** of $B$, written $A \subset B$, whenever every element of $A$ also belongs to $B$. The notation $A \subset B$ does not rule out the possibility that $A=B$. If $A \subset B$ and $A \neq B$, then we say that $A$ is a **proper subset** of $B$.
{{</definition>}}


If $A$ and $B$ are two subsets of $S$, we define

$$
A \cup B,
$$

the union of $A$ and $B$, as the set of elements of $S$ that belong to $A$ or to $B$ or to both, and

$$
A \cap B,
$$

the intersection of $A$ and $B$, as the set of elements of $S$ that belong to both $A$ and $B$. We say that $A$ and $B$ are (mutually) disjoint if they have no common elements, that is, if

$$
A \cap B=\emptyset .
$$

The complement of $A$ relative to $B$, denoted by $B-A$, is the set $\{x: x \in B$, but $x \notin A\}$. The complement of $A$ (relative to $S$) is sometimes denoted by $A^{c}$.

{{<definition "Cartesian Product" cartesian_product>}}
The **Cartesian product** of two sets $A$ and $B$, written $A \times B$, is the set of all ordered pairs $(a, b)$ such that $a \in A$ and $b \in B$. More generally, the Cartesian product of $n$ sets $A_{1}, A_{2}, \ldots, A_{n}$, written

$$
\prod_{i=1}^{n} A_{i}
$$

is the set of all ordered $n$-tuples $\left(a_{1}, a_{2}, \ldots, a_{n}\right)$ such that $a_{i} \in A_{i}(i=1, \ldots, n)$.
{{</definition>}}

The set of (finite) real numbers (the one-dimensional Euclidean space) is denoted by $\mathbb{R}$. The $n$-dimensional Euclidean space $\mathbb{R}^{n}$ is the Cartesian product of $n$ sets equal to $\mathbb{R}$:

$$
\mathbb{R}^{n}=\mathbb{R} \times \mathbb{R} \times \cdots \times \mathbb{R} \quad (n \text { times })
$$

The elements of $\mathbb{R}^{n}$ are thus the ordered $n$-tuples $\left(x_{1}, x_{2}, \ldots, x_{n}\right)$ of real numbers $x_{1}, x_{2}, \ldots, x_{n}$.

{{<definition "Bounded Set" bounded_set>}}
A set $S$ of real numbers is said to be **bounded** if there exists a number $M$ such that $|x| \leq M$ for all $x \in S$.
{{</definition>}}

## 3 - Matrices: Addition and Multiplication

{{<definition "Real Matrix" real_matrix>}}
A **real** $m \times n$ **matrix** $\mathbf{A}$ is a rectangular array of real numbers

$$
\mathbf{A}=\left(\begin{array}{cccc}
a_{11} & a_{12} & \ldots & a_{1 n} \\\\
a_{21} & a_{22} & \ldots & a_{2 n} \\\\
\vdots & \vdots & & \vdots \\\\
a_{m 1} & a_{m 2} & \ldots & a_{m n}
\end{array}\right)
$$

We sometimes write $\mathbf{A}=\left(a_{i j}\right)$.
{{</definition>}}

If one or more of the elements of $\mathbf{A}$ is complex, we say that $\mathbf{A}$ is a complex matrix. Almost all matrices in this book are real and the word 'matrix' is assumed to be a real matrix, unless explicitly stated otherwise.

An $m \times n$ matrix can be regarded as a point in $\mathbb{R}^{m \times n}$. The real numbers $a_{i j}$ are called the elements of $\mathbf{A}$. An $m \times 1$ matrix is a point in $\mathbb{R}^{m \times 1}$ (that is, in $\mathbb{R}^{m}$) and is called a (column) vector of order $m \times 1$. A $1 \times n$ matrix is called a row vector (of order $1 \times n$). The elements of a vector are usually called its components. Matrices are always denoted by capital letters and vectors by lower-case letters.

{{<definition "Matrix Addition" matrix_addition>}}
The **sum** of two matrices $\mathbf{A}$ and $\mathbf{B}$ of the same order is defined as

$$
\mathbf{A}+\mathbf{B}=\left(a_{i j}\right)+\left(b_{i j}\right)=\left(a_{i j}+b_{i j}\right)
$$
{{</definition>}}

{{<definition "Scalar Multiplication" scalar_multiplication>}}
The **product** of a matrix by a scalar $\lambda$ is

$$
\lambda \mathbf{A}=\mathbf{A} \lambda=\left(\lambda a_{i j}\right)
$$
{{</definition>}}

The following properties are now easily proved for matrices $\mathbf{A}, \mathbf{B}$, and $\mathbf{C}$ of the same order and scalars $\lambda$ and $\mu$:

\begin{equation}
\begin{aligned}
\mathbf{A}+\mathbf{B} & =\mathbf{B}+\mathbf{A}, \\\\
(\mathbf{A}+\mathbf{B})+\mathbf{C} & =\mathbf{A}+(\mathbf{B}+\mathbf{C}), \\\\
(\lambda+\mu) \mathbf{A} & =\lambda \mathbf{A}+\mu \mathbf{A}, \\\\
\lambda(\mathbf{A}+\mathbf{B}) & =\lambda \mathbf{A}+\lambda \mathbf{B}, \\\\
\lambda(\mu \mathbf{A}) & =(\lambda \mu) \mathbf{A} .
\end{aligned}
\end{equation}

A matrix whose elements are all zero is called a null matrix and denoted by $\mathbf{0}$. We have, of course,

$$
\mathbf{A}+(-1) \mathbf{A}=\mathbf{0}
$$

{{<definition "Matrix Multiplication" matrix_multiplication>}}
If $\mathbf{A}$ is an $m \times n$ matrix and $\mathbf{B}$ an $n \times p$ matrix (so that $\mathbf{A}$ has the same number of columns as $\mathbf{B}$ has rows), then we define the **product** of $\mathbf{A}$ and $\mathbf{B}$ as

$$
\mathbf{A} \mathbf{B}=\left(\sum_{j=1}^{n} a_{i j} b_{j k}\right)
$$

Thus, $\mathbf{A} \mathbf{B}$ is an $m \times p$ matrix and its $ik$th element is $\sum_{j=1}^{n} a_{i j} b_{j k}$.
{{</definition>}}

The following properties of the matrix product can be established:

\begin{equation}
\begin{aligned}
(\mathbf{A} \mathbf{B}) \mathbf{C} & =\mathbf{A}(\mathbf{B} \mathbf{C}) \\\\
\mathbf{A}(\mathbf{B}+\mathbf{C}) & =\mathbf{A} \mathbf{B}+\mathbf{A} \mathbf{C} \\\\
(\mathbf{A}+\mathbf{B}) \mathbf{C} & =\mathbf{A} \mathbf{C}+\mathbf{B} \mathbf{C}
\end{aligned}
\end{equation}

These relations hold provided the matrix products exist.

We note that the existence of $\mathbf{A} \mathbf{B}$ does not imply the existence of $\mathbf{B} \mathbf{A}$, and even when both products exist, they are not generally equal. (Two matrices $\mathbf{A}$ and $\mathbf{B}$ for which

$$
\mathbf{A} \mathbf{B}=\mathbf{B} \mathbf{A}
$$

are said to commute.) We therefore distinguish between premultiplication and postmultiplication: a given $m \times n$ matrix $\mathbf{A}$ can be premultiplied by a $p \times m$ matrix $\mathbf{B}$ to form the product $\mathbf{B} \mathbf{A}$; it can also be postmultiplied by an $n \times q$ matrix $\mathbf{C}$ to form $\mathbf{A} \mathbf{C}$.

## 4 - The transpose of a matrix

{{<definition "Transpose" transpose>}}
The **transpose** of an $m \times n$ matrix $\mathbf{A}=\left(a_{i j}\right)$ is the $n \times m$ matrix, denoted by $\mathbf{A}^{\mathrm{T}}$, whose $ij$th element is $a_{j i}$.
{{</definition>}}

We have

\begin{equation}
\begin{aligned}
\left(\mathbf{A}^{\mathrm{T}}\right)^{\mathrm{T}} & =\mathbf{A} \\\\
(\mathbf{A}+\mathbf{B})^{\mathrm{T}} & =\mathbf{A}^{\mathrm{T}}+\mathbf{B}^{\mathrm{T}} \\\\
(\mathbf{A} \mathbf{B})^{\mathrm{T}} & =\mathbf{B}^{\mathrm{T}} \mathbf{A}^{\mathrm{T}}
\end{aligned}
\end{equation}

If $\mathbf{x}$ is an $n \times 1$ vector, then $\mathbf{x}^{\mathrm{T}}$ is a $1 \times n$ row vector and

$$
\mathbf{x}^{\mathrm{T}} \mathbf{x}=\sum_{i=1}^{n} x_{i}^{2}
$$

{{<definition "Euclidean Norm" euclidean_norm>}}
The **(Euclidean) norm** of $\mathbf{x}$ is defined as

$$
\|\mathbf{x}\|=\left(\mathbf{x}^{\mathrm{T}} \mathbf{x}\right)^{1 / 2}
$$
{{</definition>}}

## 5 - Square matrices

{{<definition "Square Matrix" square_matrix>}}
A matrix is said to be **square** if it has as many rows as it has columns.
{{</definition>}}

A square matrix $\mathbf{A}=\left(a_{i j}\right)$, real or complex, is said to be

| type | if |
| :--- | :--- |
| lower triangular | if $a_{i j}=0 \quad(i<j)$, |
| strictly lower triangular | if $a_{i j}=0 \quad(i \leq j)$, |
| unit lower triangular | if $a_{i j}=0 \quad(i<j)$ and $a_{i i}=1$ (all $i$), |
| upper triangular | if $a_{i j}=0 \quad(i>j)$, |
| strictly upper triangular | if $a_{i j}=0 \quad(i \geq j)$, |
| unit upper triangular | if $a_{i j}=0 \quad(i>j)$ and $a_{i i}=1$ (all $i$), |
| idempotent | if $\mathbf{A}^{2}=\mathbf{A}$. |

A square matrix $\mathbf{A}$ is triangular if it is either lower triangular or upper triangular (or both).

A real square matrix $\mathbf{A}=\left(a_{i j}\right)$ is said to be

| type | if |
|-------|----|
| symmetric | if $\mathbf{A}^{\mathrm{T}} = \mathbf{A}$, |
| skew-symmetric | if $\mathbf{A}^{\mathrm{T}} = -\mathbf{A}$. |

For any square $n \times n$ matrix $\mathbf{A}=\left(a_{i j}\right)$, we define $\operatorname{dg} \mathbf{A}$ or $\operatorname{dg}(\mathbf{A})$ as

$$
\operatorname{dg} \mathbf{A}=\left(\begin{array}{cccc}
a_{11} & 0 & \ldots & 0 \\\\
0 & a_{22} & \ldots & 0 \\\\
\vdots & \vdots & & \vdots \\\\
0 & 0 & \ldots & a_{n n}
\end{array}\right)
$$

or, alternatively,

$$
\operatorname{dg} \mathbf{A}=\operatorname{diag}\left(a_{11}, a_{22}, \ldots, a_{n n}\right)
$$

{{<definition "Diagonal Matrix" diagonal_matrix>}}
If $\mathbf{A}=\operatorname{dg} \mathbf{A}$, we say that $\mathbf{A}$ is **diagonal**.
{{</definition>}}

{{<definition "Identity Matrix" identity_matrix>}}
A particular diagonal matrix is the **identity matrix** $\mathbf{I}_n$ (of order $n \times n$),

$$
\left (\begin{array}{cccc}
1 & 0 & \ldots & 0 \\\\
0 & 1 & \ldots & 0 \\\\
\vdots & \vdots & & \vdots \\\\
0 & 0 & \ldots & 1
\end{array}\right )=\left(\delta_{i j}\right)
$$

where $\delta_{i j}=1$ if $i=j$ and $\delta_{i j}=0$ if $i \neq j$ ($\delta_{i j}$ is called the Kronecker delta).
{{</definition>}}

We sometimes write $\mathbf{I}$ instead of $\mathbf{I}_{n}$ when the order is obvious or irrelevant. We have

$$
\mathbf{I} \mathbf{A}=\mathbf{A} \mathbf{I}=\mathbf{A},
$$

if $\mathbf{A}$ and $\mathbf{I}$ have the same order.

{{<definition "Orthogonal Matrix" orthogonal_matrix>}}
A real square matrix $\mathbf{A}$ is said to be **orthogonal** if

$$
\mathbf{A} \mathbf{A}^{\mathrm{T}}=\mathbf{A}^{\mathrm{T}} \mathbf{A}=\mathbf{I}
$$

and its columns are said to be **orthonormal**.
{{</definition>}}

A rectangular (not square) matrix can still have the property that $\mathbf{A} \mathbf{A}^{\mathrm{T}}=\mathbf{I}$ or $\mathbf{A}^{\mathrm{T}} \mathbf{A}=\mathbf{I}$, but not both. Such a matrix is called semi-orthogonal.

Note carefully that the concepts of symmetry, skew-symmetry, and orthogonality are defined only for real square matrices. Hence, a complex matrix $\mathbf{Z}$ satisfying $\mathbf{Z}^{\mathrm{T}}=\mathbf{Z}$ is not called symmetric (in spite of what some textbooks do). This is important because complex matrices can be Hermitian, skew-Hermitian, or unitary, and there are many important results about these classes of matrices. These results should specialize to matrices that are symmetric, skew-symmetric, or orthogonal in the special case that the matrices are real. Thus, a symmetric matrix is just a real Hermitian matrix, a skew-symmetric matrix is a real skew-Hermitian matrix, and an orthogonal matrix is a real unitary matrix; see also Section 1.12.

## 6  - Linear forms and quadratic forms

Let $\mathbf{a}$ be an $n \times 1$ vector, $\mathbf{A}$ an $n \times n$ matrix, and $\mathbf{B}$ an $n \times m$ matrix. The expression $\mathbf{a}^{\mathrm{T}} \mathbf{x}$ is called a linear form in $\mathbf{x}$, the expression $\mathbf{x}^{\mathrm{T}} \mathbf{A} \mathbf{x}$ is a quadratic form in $\mathbf{x}$, and the expression $\mathbf{x}^{\mathrm{T}} \mathbf{B} \mathbf{y}$ a bilinear form in $\mathbf{x}$ and $\mathbf{y}$. In quadratic forms we may, without loss of generality, assume that $\mathbf{A}$ is symmetric, because if not then we can replace $\mathbf{A}$ by $\left(\mathbf{A}+\mathbf{A}^{\mathrm{T}}\right) / 2$, since

$$
\mathbf{x}^{\mathrm{T}} \mathbf{A} \mathbf{x}=\mathbf{x}^{\mathrm{T}}\left(\frac{\mathbf{A}+\mathbf{A}^{\mathrm{T}}}{2}\right) \mathbf{x} .
$$

Thus, let $\mathbf{A}$ be a symmetric matrix. We say that $\mathbf{A}$ is

| | |
| :--- | :--- |
| positive definite | if $\mathbf{x}^{\mathrm{T}} \mathbf{A} \mathbf{x}>0$ for all $\mathbf{x} \neq \mathbf{0}$, |
| positive semidefinite | if $\mathbf{x}^{\mathrm{T}} \mathbf{A} \mathbf{x} \geq 0$ for all $\mathbf{x}$, |
| negative definite | if $\mathbf{x}^{\mathrm{T}} \mathbf{A} \mathbf{x}<0$ for all $\mathbf{x} \neq \mathbf{0}$, |
| negative semidefinite | if $\mathbf{x}^{\mathrm{T}} \mathbf{A} \mathbf{x} \leq 0$ for all $\mathbf{x}$, |
| indefinite | if $\mathbf{x}^{\mathrm{T}} \mathbf{A} \mathbf{x}>0$ for some $\mathbf{x}$ and $\mathbf{x}^{\mathrm{T}} \mathbf{A} \mathbf{x}<0$ for some $\mathbf{x}$. |

It is clear that the matrices $\mathbf{B} \mathbf{B}^{\mathrm{T}}$ and $\mathbf{B}^{\mathrm{T}} \mathbf{B}$ are positive semidefinite, and that $\mathbf{A}$ is negative (semi)definite if and only if $-\mathbf{A}$ is positive (semi)definite. A square null matrix is both positive and negative semidefinite.

{{<definition "Square Root Matrix" square_root_matrix>}}
If $\mathbf{A}$ is positive semidefinite, then there are many matrices $\mathbf{B}$ satisfying

$$
\mathbf{B}^{2}=\mathbf{A} .
$$

But there is only one positive semidefinite matrix $\mathbf{B}$ satisfying $\mathbf{B}^{2}=\mathbf{A}$. This matrix is called the **square root** of $\mathbf{A}$, denoted by $\mathbf{A}^{1 / 2}$.
{{</definition>}}

The following two theorems are often useful.

{{<theorem "Matrix Equality Conditions" matrix_equality_conditions>}}
Let $\mathbf{A}$ be an $m \times n$ matrix, $\mathbf{B}$ and $\mathbf{C}$ $n \times p$ matrices, and let $\mathbf{x}$ be an $n \times 1$ vector. Then,
(a) $\mathbf{A} \mathbf{x}=\mathbf{0} \Longleftrightarrow \mathbf{A}^{\mathrm{T}} \mathbf{A} \mathbf{x}=\mathbf{0}$,
(b) $\mathbf{A} \mathbf{B}=\mathbf{0} \Longleftrightarrow \mathbf{A}^{\mathrm{T}} \mathbf{A} \mathbf{B}=\mathbf{0}$,
(c) $\mathbf{A}^{\mathrm{T}} \mathbf{A} \mathbf{B}=\mathbf{A}^{\mathrm{T}} \mathbf{A} \mathbf{C} \Longleftrightarrow \mathbf{A} \mathbf{B}=\mathbf{A} \mathbf{C}$.
{{</theorem>}}

{{<proof>}}
(a) Clearly $\mathbf{A} \mathbf{x}=\mathbf{0}$ implies $\mathbf{A}^{\mathrm{T}} \mathbf{A} \mathbf{x}=\mathbf{0}$. Conversely, if $\mathbf{A}^{\mathrm{T}} \mathbf{A} \mathbf{x}=\mathbf{0}$, then $(\mathbf{A} \mathbf{x})^{\mathrm{T}}(\mathbf{A} \mathbf{x})=\mathbf{x}^{\mathrm{T}} \mathbf{A}^{\mathrm{T}} \mathbf{A} \mathbf{x}=0$ and hence $\mathbf{A} \mathbf{x}=\mathbf{0}$. (b) follows from (a), and (c) follows from (b) by substituting $\mathbf{B}-\mathbf{C}$ for $\mathbf{B}$ in (b).
{{</proof>}}

{{<theorem "Zero Matrix Conditions" zero_matrix_conditions>}}
Let $\mathbf{A}$ be an $m \times n$ matrix, $\mathbf{B}$ and $\mathbf{C}$ $n \times n$ matrices, $\mathbf{B}$ symmetric. Then,
(a) $\mathbf{A} \mathbf{x}=\mathbf{0}$ for all $n \times 1$ vectors $\mathbf{x}$ if and only if $\mathbf{A}=\mathbf{0}$,
(b) $\mathbf{x}^{\mathrm{T}} \mathbf{B} \mathbf{x}=0$ for all $n \times 1$ vectors $\mathbf{x}$ if and only if $\mathbf{B}=\mathbf{0}$,
(c) $\mathbf{x}^{\mathrm{T}} \mathbf{C} \mathbf{x}=0$ for all $n \times 1$ vectors $\mathbf{x}$ if and only if $\mathbf{C}^{\mathrm{T}}=-\mathbf{C}$.
{{</theorem>}}

{{<proof>}}
The proof is easy and is left to the reader.
{{</proof>}}

## 7 - The rank of a matrix

{{<definition "Linear Independence" linear_independence>}}
A set of vectors $\mathbf{x}\_{1}, \ldots, \mathbf{x}\_{n}$ 
is said to be **linearly independent** if $\sum\_{i} \alpha_{i} \mathbf{x}\_{i}=\mathbf{0}$ implies that all $\alpha\_{i}=0$. If $\mathbf{x}\_{1}, \ldots, \mathbf{x}_{n}$ are not linearly independent, they are said to be **linearly dependent**.
{{</definition>}}

{{<definition "Matrix Rank" matrix_rank>}}
Let $\mathbf{A}$ be an $m \times n$ matrix. The **column rank** of $\mathbf{A}$ is the maximum number of linearly independent columns it contains. The **row rank** of $\mathbf{A}$ is the maximum number of linearly independent rows it contains. It may be shown that the column rank of $\mathbf{A}$ is equal to its row rank. Hence, the concept of **rank** is unambiguous. We denote the rank of $\mathbf{A}$ by

$$
r(\mathbf{A}) .
$$
{{</definition>}}

It is clear that

$$
r(\mathbf{A}) \leq \min (m, n)
$$

If $r(\mathbf{A})=m$, we say that $\mathbf{A}$ has full row rank. If $r(\mathbf{A})=n$, we say that $\mathbf{A}$ has full column rank. If $r(\mathbf{A})=0$, then $\mathbf{A}$ is the null matrix, and conversely, if $\mathbf{A}$ is the null matrix, then $r(\mathbf{A})=0$.

We have the following important results concerning ranks:

\begin{equation}
\begin{gathered}
r(\mathbf{A})=r\left(\mathbf{A}^{\mathrm{T}}\right)=r\left(\mathbf{A}^{\mathrm{T}} \mathbf{A}\right)=r\left(\mathbf{A} \mathbf{A}^{\mathrm{T}}\right) \\\\
r(\mathbf{A} \mathbf{B}) \leq \min (r(\mathbf{A}), r(\mathbf{B})) \\\\
r(\mathbf{A} \mathbf{B})=r(\mathbf{A}) \quad \text { if } \mathbf{B} \text { is square and of full rank, } \\\\
r(\mathbf{A}+\mathbf{B}) \leq r(\mathbf{A})+r(\mathbf{B})
\end{gathered}
\end{equation}

and finally, if $\mathbf{A}$ is an $m \times n$ matrix and $\mathbf{A} \mathbf{x}=\mathbf{0}$ for some $\mathbf{x} \neq \mathbf{0}$, then

$$
r(\mathbf{A}) \leq n-1
$$

{{<definition "Column Space" column_space>}}
The **column space** of $\mathbf{A}(m \times n)$, denoted by $\mathcal{M}(\mathbf{A})$, is the set of vectors

$$
\mathcal{M}(\mathbf{A})=\left\\{\mathbf{y}: \mathbf{y}=\mathbf{A} \mathbf{x} \text { for some } \mathbf{x} \text { in } \mathbb{R}^{n}\right\\}
$$

Thus, $\mathcal{M}(\mathbf{A})$ is the vector space generated by the columns of $\mathbf{A}$.
{{</definition>}}

The dimension of this vector space is $r(\mathbf{A})$. We have

$
\mathcal{M}(\mathbf{A})=\mathcal{M}\left(\mathbf{A} \mathbf{A}^{\mathrm{T}}\right)
$

for any matrix $\mathbf{A}$.

**Exercises:**

1. If $\mathbf{A}$ has full column rank and $\mathbf{C}$ has full row rank, then $r(\mathbf{A} \mathbf{B} \mathbf{C})=r(\mathbf{B})$.
2. Let $\mathbf{A}$ be partitioned as $\mathbf{A}=\left(\mathbf{A}\_{1}: \mathbf{A}\_{2}\right)$. Then $r(\mathbf{A})=r\left(\mathbf{A}\_{1}\right)$ if and only if $\mathcal{M}\left(\mathbf{A}\_{2}\right) \subset \mathcal{M}\left(\mathbf{A}\_{1}\right)$.

## 8 - The Inverse

{{<definition "Nonsingular Matrix" nonsingular_matrix>}}
Let $\mathbf{A}$ be a square matrix of order $n \times n$. We say that $\mathbf{A}$ is **nonsingular** if $r(\mathbf{A})=n$, and that $\mathbf{A}$ is **singular** if $r(\mathbf{A})<n$.
{{</definition>}}

{{<definition "Matrix Inverse" matrix_inverse>}}
If $\mathbf{A}$ is nonsingular, then there exists a nonsingular matrix $\mathbf{B}$ such that

$
\mathbf{A} \mathbf{B}=\mathbf{B} \mathbf{A}=\mathbf{I}_{n} .
$

The matrix $\mathbf{B}$, denoted by $\mathbf{A}^{-1}$, is unique and is called the **inverse** of $\mathbf{A}$.
{{</definition>}}

We have

\begin{equation}
\begin{aligned}
\left(\mathbf{A}^{-1}\right)^{\mathrm{T}} & =\left(\mathbf{A}^{\mathrm{T}}\right)^{-1}, \\\\
(\mathbf{A} \mathbf{B})^{-1} & =\mathbf{B}^{-1} \mathbf{A}^{-1},
\end{aligned}
\end{equation}

if the inverses exist.

{{<definition "Permutation Matrix" permutation_matrix>}}
A square matrix $\mathbf{P}$ is said to be a **permutation matrix** if each row and each column of $\mathbf{P}$ contain a single element one, and the remaining elements are zero. An $n \times n$ permutation matrix thus contains $n$ ones and $n(n-1)$ zeros.
{{</definition>}}

It can be proved that any permutation matrix is nonsingular. In fact, it is even true that $\mathbf{P}$ is orthogonal, that is,

$
\mathbf{P}^{-1}=\mathbf{P}^{\mathrm{T}}
$

for any permutation matrix $\mathbf{P}$.

## 9 - The Determinant

{{<definition "Determinant" determinant>}}
Associated with any $n \times n$ matrix $\mathbf{A}$ is the **determinant** $|\mathbf{A}|$ defined by

$
|\mathbf{A}|=\sum(-1)^{\phi\left(j_{1}, \ldots, j_{n}\right)} \prod_{i=1}^{n} a_{i j_{i}}
$

where the summation is taken over all permutations $\left(j_{1}, \ldots, j_{n}\right)$ of the set of integers $(1, \ldots, n)$, and $\phi\left(j_{1}, \ldots, j_{n}\right)$ is the number of transpositions required to change $(1, \ldots, n)$ into $\left(j_{1}, \ldots, j_{n}\right)$.
{{</definition>}}

We have

\begin{equation}
\begin{aligned}
|\mathbf{A} \mathbf{B}| & =|\mathbf{A}||\mathbf{B}| \\\\
\left|\mathbf{A}^{\mathrm{T}}\right| & =|\mathbf{A}| \\\\
|\alpha \mathbf{A}| & =\alpha^{n}|\mathbf{A}| \quad \text { for any scalar } \alpha \\\\
\left|\mathbf{A}^{-1}\right| & =|\mathbf{A}|^{-1} \quad \text { if } \mathbf{A} \text { is nonsingular, } \\\\
\left|\mathbf{I}_{n}\right| & =1
\end{aligned}
\end{equation}

{{<definition "Minor and Cofactor" minor_cofactor>}}
A **submatrix** of $\mathbf{A}$ is the rectangular array obtained from $\mathbf{A}$ by deleting some of its rows and/or some of its columns. A **minor** is the determinant of a square submatrix of $\mathbf{A}$. The **minor** of an element $a_{i j}$ is the determinant of the submatrix of $\mathbf{A}$ obtained by deleting the $i$th row and $j$th column. The **cofactor** of $a_{i j}$, say $c_{i j}$, is $(-1)^{i+j}$ times the minor of $a_{i j}$.
{{</definition>}}

The matrix $\mathbf{C}=\left(c_{i j}\right)$ is called the cofactor matrix of $\mathbf{A}$. The transpose of $\mathbf{C}$ is called the adjoint of $\mathbf{A}$ and will be denoted by $\mathbf{A}^{\\\#}$.

We have

\begin{equation}
\begin{aligned}
|\mathbf{A}|=\sum_{j=1}^{n} a_{i j} c_{i j} & =\sum_{j=1}^{n} a_{j k} c_{j k} \quad(i, k=1, \ldots, n), \\\\
\mathbf{A} \mathbf{A}^{\\\#} & =\mathbf{A}^{\\\#} \mathbf{A}=|\mathbf{A}| \mathbf{I}, \\\\
(\mathbf{A} \mathbf{B})^{\\\#} & =\mathbf{B}^{\\\#} \mathbf{A}^{\\\#} .
\end{aligned}
\end{equation}

{{<definition "Principal Minor" principal_minor>}}
For any square matrix $\mathbf{A}$, a **principal submatrix** of $\mathbf{A}$ is obtained by deleting corresponding rows and columns. The determinant of a principal submatrix is called a **principal minor**.
{{</definition>}}

**Exercises:**

1. If $\mathbf{A}$ is nonsingular, show that $\mathbf{A}^{\\\#}=|\mathbf{A}| \mathbf{A}^{-1}$.
2. Prove that the determinant of a triangular matrix is the product of its diagonal elements.

## 10 - The trace

{{<definition "Trace" trace>}}
The **trace** of a square $n \times n$ matrix $\mathbf{A}$, denoted by $\operatorname{tr} \mathbf{A}$ or $\operatorname{tr}(\mathbf{A})$, is the sum of its diagonal elements:

$
\operatorname{tr} \mathbf{A}=\sum_{i=1}^{n} a_{i i} .
$
{{</definition>}}

We have

\begin{equation}
\begin{aligned}
\operatorname{tr}(\mathbf{A}+\mathbf{B}) & =\operatorname{tr} \mathbf{A}+\operatorname{tr} \mathbf{B} \\\\
\operatorname{tr}(\lambda \mathbf{A}) & =\lambda \operatorname{tr} \mathbf{A} \quad \text { if } \lambda \text { is a scalar } \\\\
\operatorname{tr} \mathbf{A}^{\mathrm{T}} & =\operatorname{tr} \mathbf{A} \\\\
\operatorname{tr} \mathbf{A} \mathbf{B} & =\operatorname{tr} \mathbf{B} \mathbf{A}
\end{aligned}
\end{equation}

We note in (25) that $\mathbf{A} \mathbf{B}$ and $\mathbf{B} \mathbf{A}$, though both square, need not be of the same order.

Corresponding to the vector (Euclidean) norm

$
\|\mathbf{x}\|=\left(\mathbf{x}^{\mathrm{T}} \mathbf{x}\right)^{1 / 2},
$

given in (4), we now define the matrix (Euclidean) norm as

{{<definition "Matrix Norm" matrix_norm>}}
$
\|\mathbf{A}\|=\left(\operatorname{tr} \mathbf{A}^{\mathrm{T}} \mathbf{A}\right)^{1 / 2}
$
{{</definition>}}

We have

$
\operatorname{tr} \mathbf{A}^{\mathrm{T}} \mathbf{A} \geq 0
$

with equality if and only if $\mathbf{A}=\mathbf{0}$.

## 11  - Partitioned matrices


{{<definition "Partitioned Matrix" partitioned_matrix>}}
Let $\mathbf{A}$ be an $m \times n$ matrix. We can **partition** $\mathbf{A}$ as

$$
\mathbf{A}=\left(\begin{array}{ll}
\mathbf{A}\_{11} & \mathbf{A}\_{12} \\\\
\mathbf{A}\_{21} & \mathbf{A}\_{22}
\end{array}\right),
$$

where $\mathbf{A}\_{11}$ is $m\_1 \times n\_1$, $\mathbf{A}\_{12}$ is $m\_1 \times n\_2$, $\mathbf{A}\_{21}$ is $m\_2 \times n\_1$, $\mathbf{A}\_{22}$ is $m\_2 \times n\_2$, and $m\_1+m\_2=m$ and $n\_1+n\_2=n$.
{{</definition>}}

Let $\mathbf{B}(m \times n)$ be similarly partitioned into submatrices $\mathbf{B}\_{ij}(i, j=1,2)$. Then,

$$
\mathbf{A}+\mathbf{B}=\left(\begin{array}{cc}
\mathbf{A}\_{11}+\mathbf{B}\_{11} & \mathbf{A}\_{12}+\mathbf{B}\_{12} \\\\
\mathbf{A}\_{21}+\mathbf{B}\_{21} & \mathbf{A}\_{22}+\mathbf{B}\_{22}
\end{array}\right)
$$

Now let $\mathbf{C}(n \times p)$ be partitioned into submatrices $\mathbf{C}\_{ij}(i, j=1,2)$ such that $\mathbf{C}\_{11}$ has $n\_1$ rows (and hence $\mathbf{C}\_{12}$ also has $n\_1$ rows and $\mathbf{C}\_{21}$ and $\mathbf{C}\_{22}$ have $n\_2$ rows). Then we may postmultiply $\mathbf{A}$ by $\mathbf{C}$ yielding

$$
\mathbf{A} \mathbf{C}=\left(\begin{array}{cc}
\mathbf{A}\_{11} \mathbf{C}\_{11}+\mathbf{A}\_{12} \mathbf{C}\_{21} & \mathbf{A}\_{11} \mathbf{C}\_{12}+\mathbf{A}\_{12} \mathbf{C}\_{22} \\\\
\mathbf{A}\_{21} \mathbf{C}\_{11}+\mathbf{A}\_{22} \mathbf{C}\_{21} & \mathbf{A}\_{21} \mathbf{C}\_{12}+\mathbf{A}\_{22} \mathbf{C}\_{22}
\end{array}\right)
$$

The transpose of the matrix $\mathbf{A}$ given in (28) is

$$
\mathbf{A}^{\mathrm{T}}=\left(\begin{array}{cc}
\mathbf{A}\_{11}^{\mathrm{T}} & \mathbf{A}\_{21}^{\mathrm{T}} \\\\
\mathbf{A}\_{12}^{\mathrm{T}} & \mathbf{A}\_{22}^{\mathrm{T}}
\end{array}\right)
$$

If the off-diagonal blocks $\mathbf{A}\_{12}$ and $\mathbf{A}\_{21}$ are both zero, and $\mathbf{A}\_{11}$ and $\mathbf{A}\_{22}$ are square and nonsingular, then $\mathbf{A}$ is also nonsingular and its inverse is

$$
\mathbf{A}^{-1}=\left(\begin{array}{cc}
\mathbf{A}\_{11}^{-1} & \mathbf{0} \\\\
\mathbf{0} & \mathbf{A}\_{22}^{-1}
\end{array}\right)
$$

More generally, if $\mathbf{A}$ as given in (28) is nonsingular and $\mathbf{D}=\mathbf{A}\_{22}-\mathbf{A}\_{21} \mathbf{A}\_{11}^{-1} \mathbf{A}\_{12}$ is also nonsingular, then

$$
\mathbf{A}^{-1}=\left(\begin{array}{cc}
\mathbf{A}\_{11}^{-1}+\mathbf{A}\_{11}^{-1} \mathbf{A}\_{12} \mathbf{D}^{-1} \mathbf{A}\_{21} \mathbf{A}\_{11}^{-1} & -\mathbf{A}\_{11}^{-1} \mathbf{A}\_{12} \mathbf{D}^{-1} \\\\
-\mathbf{D}^{-1} \mathbf{A}\_{21} \mathbf{A}\_{11}^{-1} & \mathbf{D}^{-1}
\end{array}\right)
$$

Alternatively, if $\mathbf{A}$ is nonsingular and $\mathbf{E}=\mathbf{A}\_{11}-\mathbf{A}\_{12} \mathbf{A}\_{22}^{-1} \mathbf{A}\_{21}$ is also nonsingular, then

$$
\mathbf{A}^{-1}=\left(\begin{array}{cc}
\mathbf{E}^{-1} & -\mathbf{E}^{-1} \mathbf{A}\_{12} \mathbf{A}\_{22}^{-1} \\\\
-\mathbf{A}\_{22}^{-1} \mathbf{A}\_{21} \mathbf{E}^{-1} & \mathbf{A}\_{22}^{-1}+\mathbf{A}\_{22}^{-1} \mathbf{A}\_{21} \mathbf{E}^{-1} \mathbf{A}\_{12} \mathbf{A}\_{22}^{-1}
\end{array}\right) .
$$

Of course, if both $\mathbf{D}$ and $\mathbf{E}$ are nonsingular, blocks in (29) and (30) can be interchanged. The results (29) and (30) can be easily extended to a $3 \times 3$ matrix partition. We only consider the following symmetric case where two of the off-diagonal blocks are null matrices.

{{<theorem "3x3 Symmetric Partitioned Matrix Inverse" symmetric_3x3_inverse>}}
If the matrix

$$
\left(\begin{array}{lll}
\mathbf{A} & \mathbf{B} & \mathbf{C} \\\\
\mathbf{B}^{\mathrm{T}} & \mathbf{D} & \mathbf{0} \\\\
\mathbf{C}^{\mathrm{T}} & \mathbf{0} & \mathbf{E}
\end{array}\right)
$$

is symmetric and nonsingular, its inverse is given by

$$
\left(\begin{array}{ccc}
\mathbf{Q}^{-1} & -\mathbf{Q}^{-1} \mathbf{B} \mathbf{D}^{-1} & -\mathbf{Q}^{-1} \mathbf{C} \mathbf{E}^{-1} \\\\
-\mathbf{D}^{-1} \mathbf{B}^{\mathrm{T}} \mathbf{Q}^{-1} & \mathbf{D}^{-1}+\mathbf{D}^{-1} \mathbf{B}^{\mathrm{T}} \mathbf{Q}^{-1} \mathbf{B} \mathbf{D}^{-1} & \mathbf{D}^{-1} \mathbf{B}^{\mathrm{T}} \mathbf{Q}^{-1} \mathbf{C} \mathbf{E}^{-1} \\\\
-\mathbf{E}^{-1} \mathbf{C}^{\mathrm{T}} \mathbf{Q}^{-1} & \mathbf{E}^{-1} \mathbf{C}^{\mathrm{T}} \mathbf{Q}^{-1} \mathbf{B} \mathbf{D}^{-1} & \mathbf{E}^{-1}+\mathbf{E}^{-1} \mathbf{C}^{\mathrm{T}} \mathbf{Q}^{-1} \mathbf{C} \mathbf{E}^{-1}
\end{array}\right)
$$

where

$$
\mathbf{Q}=\mathbf{A}-\mathbf{B} \mathbf{D}^{-1} \mathbf{B}^{\mathrm{T}}-\mathbf{C} \mathbf{E}^{-1} \mathbf{C}^{\mathrm{T}} .
$$
{{</theorem>}}

{{<proof>}}
The proof is left to the reader.
{{</proof>}}

As to the determinants of partitioned matrices, we note that

$$
\left|\begin{array}{ll}
\mathbf{A}\_{11} & \mathbf{A}\_{12} \\\\
\mathbf{0} & \mathbf{A}\_{22}
\end{array}\right|=\left|\mathbf{A}\_{11}\right|\left|\mathbf{A}\_{22}\right|=\left|\begin{array}{ll}
\mathbf{A}\_{11} & \mathbf{0} \\\\
\mathbf{A}\_{21} & \mathbf{A}\_{22}
\end{array}\right|
$$

if both $\mathbf{A}\_{11}$ and $\mathbf{A}\_{22}$ are square matrices.

**Exercises:**

1. Find the determinant and inverse (if it exists) of

$$
\mathbf{B}=\left(\begin{array}{cc}
\mathbf{A} & \mathbf{0} \\\\
\mathbf{a}^{\mathrm{T}} & 1
\end{array}\right) .
$$

2. If $|\mathbf{A}| \neq 0$, prove that

$$
\left|\begin{array}{cc}
\mathbf{A} & \mathbf{b} \\\\
\mathbf{a}^{\mathrm{T}} & \alpha
\end{array}\right|=\left(\alpha-\mathbf{a}^{\mathrm{T}} \mathbf{A}^{-1} \mathbf{b}\right)|\mathbf{A}| .
$$

3. If $\alpha \neq 0$, prove that

$$
\left|\begin{array}{cc}
\mathbf{A} & \mathbf{b} \\\\
\mathbf{a}^{\mathrm{T}} & \alpha
\end{array}\right|=\alpha\left|\mathbf{A}-(1 / \alpha) \mathbf{b} \mathbf{a}^{\mathrm{T}}\right| .
$$

## 12 - Complex Matrices

If $\mathbf{A}$ and $\mathbf{B}$ are real matrices of the same order, then a complex matrix $\mathbf{Z}$ can be defined as

$$
\mathbf{Z}=\mathbf{A}+i \mathbf{B}
$$

where $i$ denotes the imaginary unit with the property $i^2=-1$. The complex conjugate of $\mathbf{Z}$, denoted by $\mathbf{Z}^mathrm{H}$, is defined as

$$
\mathbf{Z}^\mathrm{H}=\mathbf{A}^{\mathrm{T}}-i \mathbf{B}^{\mathrm{T}}
$$

If $\mathbf{Z}$ is real, then $\mathbf{Z}^\mathrm{H}=\mathbf{Z}^{\mathrm{T}}$. If $\mathbf{Z}$ is a scalar, say $\zeta$, we usually write $\bar{\zeta}$ instead of $\zeta^mathrm{H}$.

A square complex matrix $\mathbf{Z}$ is said to be Hermitian if $\mathbf{Z}^{\mathrm{H}}=\mathbf{Z}$ (the complex equivalent to a symmetric matrix), skew-Hermitian if $\mathbf{Z}^{\mathrm{H}}=-\mathbf{Z}$ (the complex equivalent to a skew-symmetric matrix), and unitary if $\mathbf{Z}^{\mathrm{H}} \mathbf{Z}=\mathbf{I}$ (the complex equivalent to an orthogonal matrix).

We shall see in {{< theoremref symmetric_eigenvalues >}} that the eigenvalues of a symmetric matrix are real. In general, however, eigenvalues (and hence eigenvectors) are complex. In this book, complex numbers appear only in connection with eigenvalues and eigenvectors of matrices that are not symmetric (Chapter 8). A detailed treatment is therefore omitted. Matrices and vectors are assumed to be real, unless it is explicitly specified that they are complex.

## 13  - Eigenvalues and Eigenvectors

{{<definition "Eigenvalues and Eigenvectors" eigenvalues_eigenvectors>}}
Let $\mathbf{A}$ be a square matrix, say $n \times n$. The **eigenvalues** of $\mathbf{A}$ are defined as the roots of the characteristic equation

$$
\left|\lambda \mathbf{I}\_n-\mathbf{A}\right|=0
$$

The characteristic equation (31) has $n$ roots, in general complex. Let $\lambda$ be an eigenvalue of $\mathbf{A}$. Then there exist vectors $\mathbf{x}$ and $\mathbf{y}(\mathbf{x} \neq \mathbf{0}, \mathbf{y} \neq \mathbf{0})$ such that

$$
(\lambda \mathbf{I}-\mathbf{A}) \mathbf{x}=\mathbf{0}, \quad \mathbf{y}^{\mathrm{T}}(\lambda \mathbf{I}-\mathbf{A})=\mathbf{0}
$$

That is,

$$
\mathbf{A} \mathbf{x}=\lambda \mathbf{x}, \quad \mathbf{y}^{\mathrm{T}} \mathbf{A}=\lambda \mathbf{y}^{\mathrm{T}}
$$

The vectors $\mathbf{x}$ and $\mathbf{y}$ are called a **(column) eigenvector** and **row eigenvector** of $\mathbf{A}$ associated with the eigenvalue $\lambda$.
{{</definition>}}

Eigenvectors are usually normalized in some way to make them unique, for example, by $\mathbf{x}^{\mathrm{T}} \mathbf{x}=\mathbf{y}^{\mathrm{T}} \mathbf{y}=1$ (when $\mathbf{x}$ and $\mathbf{y}$ are real).

Not all roots of the characteristic equation need to be different. Each root is counted a number of times equal to its multiplicity. When a root (eigenvalue) appears more than once it is called a multiple eigenvalue; if it appears only once it is called a simple eigenvalue.

Although eigenvalues are in general complex, the eigenvalues of a symmetric matrix are always real.

{{<theorem "Symmetric Matrix Eigenvalues" symmetric_eigenvalues>}}
A symmetric matrix has only real eigenvalues.
{{</theorem>}}

{{<proof>}}
Let $\lambda$ be an eigenvalue of a symmetric matrix $\mathbf{A}$ and let $\mathbf{x}=\mathbf{u}+i \mathbf{v}$ be an associated eigenvector. Then,

$$
\mathbf{A}(\mathbf{u}+i \mathbf{v})=\lambda(\mathbf{u}+i \mathbf{v})
$$

and hence

$$
(\mathbf{u}-i \mathbf{v})^{\mathrm{T}} \mathbf{A}(\mathbf{u}+i \mathbf{v})=\lambda(\mathbf{u}-i \mathbf{v})^{\mathrm{T}}(\mathbf{u}+i \mathbf{v})
$$

which leads to

$$
\mathbf{u}^{\mathrm{T}} \mathbf{A} \mathbf{u}+\mathbf{v}^{\mathrm{T}} \mathbf{A} \mathbf{v}=\lambda\left(\mathbf{u}^{\mathrm{T}} \mathbf{u}+\mathbf{v}^{\mathrm{T}} \mathbf{v}\right)
$$

because of the symmetry of $\mathbf{A}$. This implies that $\lambda$ is real.
{{</proof>}}

Let us prove the following three results, which will be useful to us later.

{{<theorem "Similar Matrices Eigenvalues" similar_matrices_eigenvalues>}}
If $\mathbf{A}$ is an $n \times n$ matrix and $\mathbf{G}$ is a nonsingular $n \times n$ matrix, then $\mathbf{A}$ and $\mathbf{G}^{-1} \mathbf{A} \mathbf{G}$ have the same set of eigenvalues (with the same multiplicities).
{{</theorem>}}

{{<proof>}}
From

$$
\lambda \mathbf{I}\_n-\mathbf{G}^{-1} \mathbf{A} \mathbf{G}=\mathbf{G}^{-1}\left(\lambda \mathbf{I}\_n-\mathbf{A}\right) \mathbf{G}
$$

we obtain

$$
\left|\lambda \mathbf{I}\_n-\mathbf{G}^{-1} \mathbf{A} \mathbf{G}\right|=\left|\mathbf{G}^{-1}\right|\left|\lambda \mathbf{I}\_n-\mathbf{A}\right||\mathbf{G}|=\left|\lambda \mathbf{I}\_n-\mathbf{A}\right|
$$

and the result follows.
{{</proof>}}

{{<theorem "Singular Matrix Zero Eigenvalue" singular_zero_eigenvalue>}}
A singular matrix has at least one zero eigenvalue.
{{</theorem>}}

{{<proof>}}
If $\mathbf{A}$ is singular, then $|\mathbf{A}|=0$ and hence $|\lambda \mathbf{I}-\mathbf{A}|=0$ for $\lambda=0$.
{{</proof>}}

{{<theorem "Special Matrix Eigenvalues" special_matrix_eigenvalues>}}
An idempotent matrix has only eigenvalues 0 or 1. All eigenvalues of a unitary matrix have unit modulus.
{{</theorem>}}

{{<proof>}}
Let $\mathbf{A}$ be idempotent. Then $\mathbf{A}^2=\mathbf{A}$. Thus, if $\mathbf{A} \mathbf{x}=\lambda \mathbf{x}$, then

$$
\lambda \mathbf{x}=\mathbf{A} \mathbf{x}=\mathbf{A}(\mathbf{A} \mathbf{x})=\mathbf{A}(\lambda \mathbf{x})=\lambda(\mathbf{A} \mathbf{x})=\lambda^2 \mathbf{x}
$$

and hence $\lambda=\lambda^2$, which implies $\lambda=0$ or $\lambda=1$.

If $\mathbf{A}$ is unitary, then $\mathbf{A}^{\mathrm{H}} \mathbf{A}=\mathbf{I}$. Thus, if $\mathbf{A} \mathbf{x}=\lambda \mathbf{x}$, then

$$
\mathbf{x}^{\mathrm{H}} \mathbf{A}^{\mathrm{H}}=\bar{\lambda} \mathbf{x}^{\mathrm{H}}
$$

using the notation of Section 1.12. Hence,

$$
\mathbf{x}^{\mathrm{H}} \mathbf{x}=\mathbf{x}^{\mathrm{H}} \mathbf{A}^{\mathrm{H}} \mathbf{A} \mathbf{x}=\bar{\lambda} \lambda \mathbf{x}^{\mathrm{H}} \mathbf{x}
$$

Since $\mathbf{x}^{\mathrm{H}} \mathbf{x} \neq 0$, we obtain $\bar{\lambda} \lambda=1$ and hence $|\lambda|=1$.
{{</proof>}}

An important theorem regarding positive definite matrices is stated below.

{{<theorem "Positive Definite Eigenvalues" positive_definite_eigenvalues>}}
A symmetric matrix is positive definite if and only if all its eigenvalues are positive.
{{</theorem>}}

{{<proof>}}
If $\mathbf{A}$ is positive definite and $\mathbf{A}\mathbf{x}=\lambda \mathbf{x}$, then $\mathbf{x}^\mathrm{T} \mathbf{A} \mathbf{x}=\lambda \mathbf{x}^\mathrm{T} \mathbf{x}$. Now, $\mathbf{x}^\mathrm{T} \mathbf{A} \mathbf{x}>0$ and $\mathbf{x}^\mathrm{T} \mathbf{x}>0$ imply $\lambda>0$. The converse will not be proved here. (It follows from {{<theoremref schur_decomposition>}}.)
{{</proof>}}

Next, let us prove {{<theoremref eigenvalue_identity>}}.

{{<theorem "Eigenvalue Identity" eigenvalue_identity>}}
Let $\mathbf{A}$ be $m \times n$ and let $\mathbf{B}$ be $n \times m(n \geq m)$. Then the nonzero eigenvalues of $\mathbf{B}\mathbf{A}$ and $\mathbf{A}\mathbf{B}$ are identical, and $\left|I_{m}-\mathbf{A}\mathbf{B}\right|=\left|I_{n}-\mathbf{B}\mathbf{A}\right|$.
{{</theorem>}}

{{<proof>}}
Taking determinants on both sides of the equality

\begin{equation}
\left(\begin{array}{cc}
I_{m}-\mathbf{A}\mathbf{B} & \mathbf{A} \\\\
0 & I_{n}
\end{array}\right)\left(\begin{array}{cc}
I_{m} & 0 \\\\
\mathbf{B} & I_{n}
\end{array}\right)=\left(\begin{array}{cc}
I_{m} & 0 \\\\
\mathbf{B} & I_{n}
\end{array}\right)\left(\begin{array}{cc}
I_{m} & \mathbf{A} \\\\
0 & I_{n}-\mathbf{B}\mathbf{A}
\end{array}\right),
\end{equation}

we obtain

\begin{equation}
\left|I_{m}-\mathbf{A}\mathbf{B}\right|=\left|I_{n}-\mathbf{B}\mathbf{A}\right| .
\end{equation}

Now let $\lambda \neq 0$. Then,

\begin{equation}
\begin{aligned}
\left|\lambda I_{n}-\mathbf{B}\mathbf{A}\right| & =\lambda^{n}\left|I_{n}-\mathbf{B}\left(\lambda^{-1} \mathbf{A}\right)\right| \\\\
& =\lambda^{n}\left|I_{m}-\left(\lambda^{-1} \mathbf{A}\right) \mathbf{B}\right|=\lambda^{n-m}\left|\lambda I_{m}-\mathbf{A}\mathbf{B}\right| .
\end{aligned}
\end{equation}

Hence, the nonzero eigenvalues of $\mathbf{B}\mathbf{A}$ are the same as the nonzero eigenvalues of $\mathbf{A}\mathbf{B}$, and this is equivalent to the statement in the theorem.
{{</proof>}}

Without proof we state the following famous result.

{{<theorem "Cayley-Hamilton" cayley_hamilton>}}
Let $\mathbf{A}$ be an $n \times n$ matrix with eigenvalues $\lambda_{1}, \ldots, \lambda_{n}$. Then,

\begin{equation}
\prod_{i=1}^{n}\left(\lambda_{i} I_{n}-\mathbf{A}\right)=0 .
\end{equation}
{{</theorem>}}

Finally, we present the following result on eigenvectors.

{{<theorem "Linear Independence of Eigenvectors" eigenvector_independence>}}
Eigenvectors associated with distinct eigenvalues are linearly independent.
{{</theorem>}}

{{<proof>}}
Let $\mathbf{A}\mathbf{x}\_{1}=\lambda\_{1} \mathbf{x}\_{1}$, $\mathbf{A}\mathbf{x}\_{2}=\lambda_{2} \mathbf{x}\_{2}$, and $\lambda\_{1} \neq \lambda\_{2}$. Assume that $\mathbf{x}\_{1}$ and $\mathbf{x}\_{2}$ are linearly dependent. Then there is an $\alpha \neq 0$ such that $\mathbf{x}_{2}=\alpha \mathbf{x}\_{1}$, and hence

\begin{equation}
\alpha \lambda\_{1} \mathbf{x}\_{1}=\alpha \mathbf{A} \mathbf{x}\_{1}=\mathbf{A} \mathbf{x}\_{2}=\lambda\_{2} \mathbf{x}\_{2}=\alpha \lambda\_{2} \mathbf{x}\_{1}
\end{equation}

that is

\begin{equation}
\alpha\left(\lambda_{1}-\lambda_{2}\right) \mathbf{x}_{1}=0
\end{equation}

Since $\alpha \neq 0$ and $\lambda_{1} \neq \lambda_{2}$, this implies that $\mathbf{x}_{1}=0$, a contradiction.
{{</proof>}}

**Exercices**

1. Show that

\begin{equation}
\left|\begin{array}{ll}
0 & I_{m} \\\\
I_{m} & 0
\end{array}\right|=(-1)^{m}
\end{equation}

2. Show that, for $n=2$,

\begin{equation}
|I+\epsilon \mathbf{A}|=1+\epsilon \operatorname{tr} \mathbf{A}+\epsilon^{2}|\mathbf{A}|
\end{equation}

3. Show that, for $n=3$,

\begin{equation}
|I+\epsilon \mathbf{A}|=1+\epsilon \operatorname{tr} \mathbf{A}+\frac{\epsilon^{2}}{2}\left((\operatorname{tr} \mathbf{A})^{2}-\operatorname{tr} \mathbf{A}^{2}\right)+\epsilon^{3}|\mathbf{A}|
\end{equation}


## 14 - Schur's decomposition theorem
In the next few sections, we present three decomposition theorems: Schur's theorem, Jordan's theorem, and the singular-value decomposition. Each of these theorems will prove useful later in this book. We first state Schur's theorem.

{{<theorem "Schur decomposition" schur_decomposition>}}
Let $\mathbf{A}$ be an $n \times n$ matrix, possibly complex. Then there exist a unitary $n \times n$ matrix $\mathbf{S}$ (that is, $\mathbf{S}^\mathrm{H} \mathbf{S}=I_{n}$ ) and an upper triangular matrix $\mathbf{M}$ whose diagonal elements are the eigenvalues of $\mathbf{A}$, such that

\begin{equation}
\mathbf{S}^\mathrm{H} \mathbf{A} \mathbf{S}=\mathbf{M}
\end{equation}
{{</theorem>}}

The most important special case of Schur's decomposition theorem is the case where $\mathbf{A}$ is symmetric.

{{<theorem "Symmetric Matrix Decomposition" symmetric_decomposition>}}
Let $\mathbf{A}$ be a symmetric $n \times n$ matrix. Then there exist an orthogonal $n \times n$ matrix $\mathbf{S}$ (that is, $\mathbf{S}^\mathrm{T} \mathbf{S}=I_{n}$ ) whose columns are eigenvectors of $\mathbf{A}$ and a diagonal matrix $\boldsymbol{\Lambda}$ whose diagonal elements are the eigenvalues of $\mathbf{A}$, such that

\begin{equation}
\mathbf{S}^\mathrm{T} \mathbf{A} \mathbf{S}=\boldsymbol{\Lambda}
\end{equation}
{{</theorem>}}

{{<proof>}}
Using {{<theoremref schur_decomposition>}}, there exists a unitary matrix $\mathbf{S}=\mathbf{R}+i \mathbf{T}$ with real $\mathbf{R}$ and $\mathbf{T}$ and an upper triangular matrix $\mathbf{M}$ such that $\mathbf{S}^\mathrm{H} \mathbf{A} \mathbf{S}=\mathbf{M}$. Then,

\begin{equation}
\begin{aligned}
\mathbf{M} & =\mathbf{S}^\mathrm{H} \mathbf{A} \mathbf{S}=(\mathbf{R}-i \mathbf{T})^\mathrm{T} \mathbf{A}(\mathbf{R}+i \mathbf{T}) \\\\
& =\left(\mathbf{R}^\mathrm{T} \mathbf{A} \mathbf{R}+\mathbf{T}^\mathrm{T} \mathbf{A} \mathbf{T}\right)+i\left(\mathbf{R}^\mathrm{T} \mathbf{A} \mathbf{T}-\mathbf{T}^\mathrm{T} \mathbf{A} \mathbf{R}\right)
\end{aligned}
\end{equation}

and hence, using the symmetry of $\mathbf{A}$,

\begin{equation}
\mathbf{M}+\mathbf{M}^\mathrm{T}=2\left(\mathbf{R}^\mathrm{T} \mathbf{A} \mathbf{R}+\mathbf{T}^\mathrm{T} \mathbf{A} \mathbf{T}\right) .
\end{equation}

It follows that $\mathbf{M}+\mathbf{M}^\mathrm{T}$ is a real matrix and hence, since $\mathbf{M}$ is triangular, that $\mathbf{M}$ is a real matrix. We thus obtain, from (32),

\begin{equation}
\mathbf{M}=\mathbf{R}^\mathrm{T} \mathbf{A} \mathbf{R}+\mathbf{T}^\mathrm{T} \mathbf{A} \mathbf{T} .
\end{equation}

Since $\mathbf{A}$ is symmetric, $\mathbf{M}$ is symmetric. But, since $\mathbf{M}$ is also triangular, $\mathbf{M}$ must be diagonal. The columns of $\mathbf{S}$ are then eigenvectors of $\mathbf{A}$, and since the diagonal elements of $\mathbf{M}$ are real, $\mathbf{S}$ can be chosen to be real as well.
{{</proof>}}

**Exercices**

1. Let $\mathbf{A}$ be a symmetric $n \times n$ matrix with eigenvalues $\lambda_{1} \leq \lambda_{2} \leq \cdots \leq \lambda_{n}$. Use {{<theoremref symmetric_decomposition>}} to prove that

\begin{equation}
\lambda_{1} \leq \frac{\mathbf{x}^\mathrm{T} \mathbf{A} \mathbf{x}}{\mathbf{x}^\mathrm{T} \mathbf{x}} \leq \lambda_{n} .
\end{equation}

2. Hence show that, for any $m \times n$ matrix $\mathbf{A}$,

\begin{equation}
\|\mathbf{A} \mathbf{x}\| \leq \mu\|\mathbf{x}\|,
\end{equation}

where $\mu^{2}$ denotes the largest eigenvalue of $\mathbf{A}^\mathrm{T} \mathbf{A}$.

3. Let $\mathbf{A}$ be an $m \times n$ matrix of rank $r$. Show that there exists an $n \times(n-r)$ matrix $\mathbf{S}$ such that

\begin{equation}
\mathbf{A} \mathbf{S}=0, \quad \mathbf{S}^\mathrm{T} \mathbf{S}=I_{n-r}
\end{equation}

4. Let $\mathbf{A}$ be an $m \times n$ matrix of rank $r$. Let $\mathbf{S}$ be a matrix such that $\mathbf{A} \mathbf{S}=0$. Show that $r(\mathbf{S}) \leq n-r$.

## 15 - The Jordan decomposition

Schur's theorem tells us that there exists, for every square matrix $\mathbf{A}$, a unitary (possibly orthogonal) matrix $\mathbf{S}$ which 'transforms' $\mathbf{A}$ into an upper triangular matrix $\mathbf{M}$ whose diagonal elements are the eigenvalues of $\mathbf{A}$.

Jordan's theorem similarly states that there exists a nonsingular matrix, say $\mathbf{T}$, which transforms $\mathbf{A}$ into an upper triangular matrix $\mathbf{M}$ whose diagonal elements are the eigenvalues of $\mathbf{A}$. The difference between the two decomposition theorems is that in Jordan's theorem less structure is put on the matrix $\mathbf{T}$ (nonsingular, but not necessarily unitary) and more structure on the matrix $\mathbf{M}$.

{{<theorem "Jordan decomposition" jordan_decomposition>}}
Let $\mathbf{A}$ be an $n \times n$ matrix and denote by $\mathbf{J}_{k}(\lambda)$ a $k \times k$ matrix of the form

\begin{equation}
\mathbf{J}_{k}(\lambda)=\left(\begin{array}{cccccc}
\lambda & 1 & 0 & \ldots & 0 & 0 \\\\
0 & \lambda & 1 & \ldots & 0 & 0 \\\\
\vdots & \vdots & \vdots & & \vdots & \vdots \\\\
0 & 0 & 0 & \ldots & \lambda & 1 \\\\
0 & 0 & 0 & \ldots & 0 & \lambda
\end{array}\right)
\end{equation}

(a so-called Jordan block), where $\mathbf{J}_{1}(\lambda)=\lambda$. Then there exists a nonsingular $n \times n$ matrix $\mathbf{T}$ such that

\begin{equation}
\mathbf{T}^{-1} \mathbf{A} \mathbf{T}=\left(\begin{array}{cccc}
\mathbf{J}\_{k\_{1}}\left(\lambda\_{1}\right) & 0 & \ldots & 0 \\\\
0 & \mathbf{J}\_{k\_{2}}\left(\lambda\_{2}\right) & \ldots & 0 \\\\
\vdots & \vdots & & \vdots \\\\
0 & 0 & \ldots & \mathbf{J}\_{k\_{r}}\left(\lambda\_{r}\right)
\end{array}\right)
\end{equation}

with $k_{1}+k_{2}+\cdots+k_{r}=n$. The $\lambda_{i}$ are the eigenvalues of $\mathbf{A}$, not necessarily distinct.
{{</theorem>}}

The most important special case of {{<theoremref jordan_decomposition>}} is {{<theoremref distinct_eigenvalues>}}.

{{<theorem "Distinct Eigenvalues Decomposition" distinct_eigenvalues>}}
Let $\mathbf{A}$ be an $n \times n$ matrix with distinct eigenvalues. Then there exist a nonsingular $n \times n$ matrix $\mathbf{T}$ and a diagonal $n \times n$ matrix $\boldsymbol{\Lambda}$ whose diagonal elements are the eigenvalues of $\mathbf{A}$, such that

\begin{equation}
\mathbf{T}^{-1} \mathbf{A} \mathbf{T}=\boldsymbol{\Lambda}
\end{equation}
{{</theorem>}}

{{<proof>}}
Immediate from {{<theoremref jordan_decomposition>}} (or {{<theoremref eigenvector_independence>}}).
{{</proof>}}

**Exercices**

1. Show that $\left(\lambda I_{k}-\mathbf{J}_{k}(\lambda)\right)^{k}=0$ and use this fact to prove {{<theoremref cayley_hamilton>}}.
2. Show that {{<theoremref distinct_eigenvalues>}} remains valid when $\mathbf{A}$ is complex.

## 16 - The singular value decomposition

The third important decomposition theorem is the singular-value decomposition.

{{<theorem "Singular-value decomposition" svd>}}
Let $\mathbf{A}$ be a real $m \times n$ matrix with $r(\mathbf{A})=r>0$. Then there exist an $m \times r$ matrix $\mathbf{S}$ such that $\mathbf{S}^\mathrm{T} \mathbf{S}=I_{r}$, an $n \times r$ matrix $\mathbf{T}$ such that $\mathbf{T}^\mathrm{T} \mathbf{T}=I_{r}$ and an $r \times r$ diagonal matrix $\boldsymbol{\Lambda}$ with positive diagonal elements, such that

\begin{equation}
\mathbf{A}=\mathbf{S} \boldsymbol{\Lambda}^{1 / 2} \mathbf{T}^\mathrm{T}
\end{equation}
{{</theorem>}}

{{<proof>}}
Since $\mathbf{A} \mathbf{A}^\mathrm{T}$ is an $m \times m$ positive semidefinite matrix of rank $r$ (by (6)), its nonzero eigenvalues are all positive ({{<theoremref eigenvalue_positive>}}). From {{<theoremref symmetric_decomposition>}} we know that there exists an orthogonal $m \times m$ matrix $( \mathbf{S}: \mathbf{S}_{2})$ such that

\begin{equation}
\mathbf{A} \mathbf{A}^\mathrm{T} \mathbf{S}=\mathbf{S} \boldsymbol{\Lambda}, \quad \mathbf{A} \mathbf{A}^\mathrm{T} \mathbf{S}\_{2}=0, \quad \mathbf{S} \mathbf{S}^\mathrm{T}+\mathbf{S}\_{2} \mathbf{S}\_{2}^\mathrm{T}=I\_{m}
\end{equation}

where $\boldsymbol{\Lambda}$ is an $r \times r$ diagonal matrix having these $r$ positive eigenvalues as its diagonal elements. Define $\mathbf{T}=\mathbf{A}^\mathrm{T} \mathbf{S} \boldsymbol{\Lambda}^{-1 / 2}$. Then we see that

\begin{equation}
\mathbf{A}^\mathrm{T} \mathbf{A} \mathbf{T}=\mathbf{T} \boldsymbol{\Lambda}, \quad \mathbf{T}^\mathrm{T} \mathbf{T}=I\_{r}
\label{eq:33}
\end{equation}

Thus, since  $\mathbf{A}^\mathrm{T} \mathbf{S}\_{2}=0$, we have

\begin{equation}
\mathbf{A}=\left(\mathbf{S} \mathbf{S}^\mathrm{T}+\mathbf{S}\_{2} \mathbf{S}_{2}^\mathrm{T}\right) \mathbf{A}=\mathbf{S} \mathbf{S}^\mathrm{T} \mathbf{A}=\mathbf{S} \boldsymbol{\Lambda}^{1 / 2}\left(\mathbf{A}^\mathrm{T} \mathbf{S} \boldsymbol{\Lambda}^{-1 / 2}\right)^\mathrm{T}=\mathbf{S} \boldsymbol{\Lambda}^{1 / 2} \mathbf{T}^\mathrm{T}
\label{eq:34}
\end{equation}

which concludes the proof.
{{</proof>}}

We see from \eqref{eq:33} and \eqref{eq:34} that the semi-orthogonal matrices $\mathbf{S}$ and $\mathbf{T}$ satisfy

\begin{equation}
\mathbf{A} \mathbf{A}^\mathrm{T} \mathbf{S}=\mathbf{S} \boldsymbol{\Lambda}, \quad \mathbf{A}^\mathrm{T} \mathbf{A} \mathbf{T}=\mathbf{T} \boldsymbol{\Lambda}
\end{equation}

Hence, $\boldsymbol{\Lambda}$ contains the $r$ nonzero eigenvalues of $\mathbf{A} \mathbf{A}^\mathrm{T}$ (and of $\mathbf{A}^\mathrm{T} \mathbf{A}$ ) and $\mathbf{S}$ (by construction) and $\mathbf{T}$ contain corresponding eigenvectors. A common mistake in applying the singular-value decomposition is to find $\mathbf{S}$, $\mathbf{T}$, and $\boldsymbol{\Lambda}$ from (35). This is incorrect because, given $\mathbf{S}$, $\mathbf{T}$ is not unique. The correct procedure is to find $\mathbf{S}$ and $\boldsymbol{\Lambda}$ from $\mathbf{A} \mathbf{A}^\mathrm{T} \mathbf{S}=\mathbf{S} \boldsymbol{\Lambda}$ and then define $\mathbf{T}=\mathbf{A}^\mathrm{T} \mathbf{S} \boldsymbol{\Lambda}^{-1 / 2}$. Alternatively, we can find $\mathbf{T}$ and $\boldsymbol{\Lambda}$ from $\mathbf{A}^\mathrm{T} \mathbf{A} \mathbf{T}=\mathbf{T} \boldsymbol{\Lambda}$ and define $\mathbf{S}=\mathbf{A} \mathbf{T} \boldsymbol{\Lambda}^{-1 / 2}$.

## 17 - Further results concerning eigenvalues

Let us now prove the following five theorems, all of which concern eigenvalues. {{<theoremref trace_determinant>}} deals with the sum and the product of the eigenvalues. {{<theoremref rank_eigenvalues>}} and {{<theoremref rank_eigenvalues_symmetric>}} discuss the relationship between the rank and the number of nonzero eigenvalues, and {{<theoremref idempotent_eigenvalues>}} concerns idempotent matrices.

{{<theorem "Trace and Determinant" trace_determinant>}}
Let $\mathbf{A}$ be a square, possibly complex, $n \times n$ matrix with eigenvalues $\lambda_{1}, \ldots, \lambda_{n}$. Then,

\begin{equation}
\operatorname{tr} \mathbf{A}=\sum_{i=1}^{n} \lambda_{i}, \quad |\mathbf{A}|=\prod_{i=1}^{n} \lambda_{i}
\end{equation}
{{</theorem>}}

{{<proof>}}
We write, using {{<theoremref schur_decomposition>}}, $\mathbf{S}^\mathrm{H} \mathbf{A} \mathbf{S}=\mathbf{M}$. Then,

\begin{equation}
\operatorname{tr} \mathbf{A}=\operatorname{tr} \mathbf{S} \mathbf{M} \mathbf{S}^\mathrm{H}=\operatorname{tr} \mathbf{M} \mathbf{S}^\mathrm{H} \mathbf{S}=\operatorname{tr} \mathbf{M}=\sum\_{i} \lambda\_{i}
\end{equation}

and

\begin{equation}
|\mathbf{A}|=\left|\mathbf{S} \mathbf{M} \mathbf{S}^\mathrm{H}\right|=|\mathbf{S}||\mathbf{M}|\left|\mathbf{S}^\mathrm{H}\right|=|\mathbf{M}|=\prod_{i} \lambda_{i}
\end{equation}

and the result follows.
{{</proof>}}

{{<theorem "Rank and Nonzero Eigenvalues" rank_eigenvalues>}}
If $\mathbf{A}$ has $r$ nonzero eigenvalues, then $r(\mathbf{A}) \geq r$.
{{</theorem>}}

{{<proof>}}
We write again, using {{<theoremref schur_decomposition>}}, $\mathbf{S}^\mathrm{H} \mathbf{A} \mathbf{S}=\mathbf{M}$. We partition

\begin{equation}
\mathbf{M}=\left(\begin{array}{cc}
\mathbf{M}\_{1} & \mathbf{M}\_{2} \\\\
0 & \mathbf{M}\_{3}
\end{array}\right)
\end{equation}

where $\mathbf{M}_{1}$ is a nonsingular upper triangular $r \times r$ matrix and $\mathbf{M}\_{3}$ is strictly upper triangular. Since $r(\mathbf{A})=r(\mathbf{M}) \geq r\left(\mathbf{M}\_{1}\right)=r$, the result follows.
{{</proof>}}

The following example shows that it is indeed possible that $r(\mathbf{A})>r$. Let

\begin{equation}
\mathbf{A}=\left(\begin{array}{ll}
1 & -1 \\\\
1 & -1
\end{array}\right)
\end{equation}

Then $r(\mathbf{A})=1$ and both eigenvalues of $\mathbf{A}$ are zero.

{{<theorem "Simple Eigenvalue Rank" simple_eigenvalue_rank>}}
Let $\mathbf{A}$ be an $n \times n$ matrix. If $\lambda$ is a simple eigenvalue of $\mathbf{A}$, then $r(\lambda I-\mathbf{A})=n-1$. Conversely, if $r(\lambda I-\mathbf{A})=n-1$, then $\lambda$ is an eigenvalue of $\mathbf{A}$, but not necessarily a simple eigenvalue.
{{</theorem>}}

{{<proof>}}
Let $\lambda_{1}, \ldots, \lambda_{n}$ be the eigenvalues of $\mathbf{A}$. Then $\mathbf{B}=\lambda I-\mathbf{A}$ has eigenvalues $\lambda-\lambda_{i}(i=1, \ldots, n)$, and since $\lambda$ is a simple eigenvalue of $\mathbf{A}$, $\mathbf{B}$ has a simple eigenvalue zero. Hence, $r(\mathbf{B}) \leq n-1$. Also, since $\mathbf{B}$ has $n-1$ nonzero eigenvalues, $r(\mathbf{B}) \geq n-1$ ({{<theoremref rank_eigenvalues>}}). Hence $r(\mathbf{B})=n-1$. Conversely, if $r(\mathbf{B})=n-1$, then $\mathbf{B}$ has at least one zero eigenvalue and hence $\lambda=\lambda_{i}$ for at least one $i$.
{{</proof>}}

{{<definition "Simple Zero Eigenvalue Corollary" simple_zero_corollary>}}
An $n \times n$ matrix with a simple zero eigenvalue has rank $n-1$.
{{</definition>}}

{{<theorem "Symmetric Matrix Rank and Eigenvalues" rank_eigenvalues_symmetric>}}
If $\mathbf{A}$ is a symmetric matrix with $r$ nonzero eigenvalues, then $r(\mathbf{A})=r$.
{{</theorem>}}

{{<proof>}}
Using {{<theoremref symmetric_decomposition>}}, we have $\mathbf{S}^\mathrm{T} \mathbf{A} \mathbf{S}=\boldsymbol{\Lambda}$ and hence

\begin{equation}
r(\mathbf{A})=r\left(\mathbf{S} \boldsymbol{\Lambda} \mathbf{S}^\mathrm{T}\right)=r(\boldsymbol{\Lambda})=r,
\end{equation}

and the result follows.
{{</proof>}}

{{<theorem "Idempotent Matrix Properties" idempotent_1_12>}}
If $\mathbf{A}$ is an idempotent matrix, possibly complex, with $r$ eigenvalues equal to one, then $r(\mathbf{A})=\operatorname{tr} \mathbf{A}=r$.
{{</theorem>}}
{{<proof>}}
By {{<theoremref theorem_1_12>}}, $\mathbf{S}^{*} \mathbf{A} \mathbf{S}=\mathbf{M}$ (upper triangular), where

\begin{equation}
\mathbf{M}=\left(\begin{array}{cc}
\mathbf{M}\_{1} & \mathbf{M}\_{2} \\\\
0 & \mathbf{M}_{3}
\end{array}\right)
\end{equation}

with $\mathbf{M}\_{1}$ a unit upper triangular $r \times r$ matrix and $\mathbf{M}\_{3}$ a strictly upper triangular matrix. Since $\mathbf{A}$ is idempotent, so is $\mathbf{M}$ and hence

\begin{equation}
\left(\begin{array}{cc}
\mathbf{M}\_{1}^{2} & \mathbf{M}\_{1} \mathbf{M}_{2}+\mathbf{M}\_{2} \mathbf{M}\_{3} \\\\
0 & \mathbf{M}\_{3}^{2}
\end{array}\right)=\left(\begin{array}{cc}
\mathbf{M}\_{1} & \mathbf{M}\_{2} \\\\
0 & \mathbf{M}\_{3}
\end{array}\right) .
\end{equation}

This implies that $\mathbf{M}_{1}$ is idempotent; it is nonsingular, hence $\mathbf{M}\_{1}=\mathbf{I}\_{r}$ (see Exercise 1 below). Also, $\mathbf{M}\_{3}$ is idempotent and all its eigenvalues are zero, hence $\mathbf{M}\_{3}=0$ (see Exercise 2 below), so that

\begin{equation}
\mathbf{M}=\left(\begin{array}{cc}
\mathbf{I}_{r} & \mathbf{M}\_{2} \\\\
0 & 0
\end{array}\right)
\end{equation}

Hence,

\begin{equation}
r(\mathbf{A})=r(\mathbf{M})=r
\end{equation}

Also, by:

\begin{equation}
\operatorname{tr} \mathbf{A}=\text { sum of eigenvalues of } \mathbf{A}=r,
\end{equation}

thus completing the proof.
{{</proof>}}

We note that in, the matrix $\mathbf{A}$ is not required to be symmetric. If $\mathbf{A}$ is idempotent and symmetric, then it is positive semidefinite. Since its eigenvalues are only 0 and 1 and its rank equals $r$, it that $\mathbf{A}$ can be written as

\begin{equation}
\mathbf{A}=\mathbf{P} \mathbf{P}^{\mathrm{T}}, \quad \mathbf{P}^{\mathrm{T}} \mathbf{P}=\mathbf{I}_{r}
\end{equation}

**Exercises**

1. The only nonsingular idempotent matrix is the identity matrix.
2. The only idempotent matrix whose eigenvalues are all zero is the null matrix.
3. If $\mathbf{A}$ is a positive semidefinite $n \times n$ matrix with $r(\mathbf{A})=r$, then there exists an $n \times r$ matrix $\mathbf{P}$ such that

\begin{equation}
\mathbf{A}=\mathbf{P} \mathbf{P}^{\mathrm{T}}, \quad \mathbf{P}^{\mathrm{T}} \mathbf{P}=\mathbf{\Lambda}
\end{equation}

where $\mathbf{\Lambda}$ is an $r \times r$ diagonal matrix containing the positive eigenvalues of $\mathbf{A}$.

## Positive (semi)definite matrices

Positive (semi)definite matrices were introduced in Section 1.6. We have already seen that $\mathbf{A} \mathbf{A}^{\mathrm{T}}$ and $\mathbf{A}^{\mathrm{T}} \mathbf{A}$ are both positive semidefinite and that the eigenvalues of a positive (semi)definite matrix are all positive (nonnegative). We now present some more properties of positive (semi)definite matrices.

{{<theorem "Determinant inequality for positive definite matrices" theorem_1_22>}}
Let $\mathbf{A}$ be positive definite and $\mathbf{B}$ positive semidefinite. Then,

\begin{equation}
|\mathbf{A}+\mathbf{B}| \geq|\mathbf{A}|
\end{equation}

with equality if and only if $\mathbf{B}=0$.
{{</theorem>}}

{{<proof>}}
Let $\mathbf{\Lambda}$ be a positive definite diagonal matrix such that

\begin{equation}
\mathbf{S}^{\mathrm{T}} \mathbf{A} \mathbf{S}=\mathbf{\Lambda}, \quad \mathbf{S}^{\mathrm{T}} \mathbf{S}=\mathbf{I} .
\end{equation}

Then, $\mathbf{S} \mathbf{S}^{\mathrm{T}}=\mathbf{I}$ and

\begin{equation}
\mathbf{A}+\mathbf{B}=\mathbf{S} \mathbf{\Lambda}^{1 / 2}\left(\mathbf{I}+\mathbf{\Lambda}^{-1 / 2} \mathbf{S}^{\mathrm{T}} \mathbf{B} \mathbf{S} \mathbf{\Lambda}^{-1 / 2}\right) \mathbf{\Lambda}^{1 / 2} \mathbf{S}^{\mathrm{T}} .
\end{equation}

Hence, using determinant results,

\begin{equation}
\begin{aligned}
|\mathbf{A}+\mathbf{B}| & =\left|\mathbf{S} \mathbf{\Lambda}^{1 / 2}\right|\left|\mathbf{I}+\mathbf{\Lambda}^{-1 / 2} \mathbf{S}^{\mathrm{T}} \mathbf{B} \mathbf{S} \mathbf{\Lambda}^{-1 / 2}\right|\left|\mathbf{\Lambda}^{1 / 2} \mathbf{S}^{\mathrm{T}}\right| \\\\
& =\left|\mathbf{S} \mathbf{\Lambda}^{1 / 2} \mathbf{\Lambda}^{1 / 2} \mathbf{S}^{\mathrm{T}}\right|\left|\mathbf{I}+\mathbf{\Lambda}^{-1 / 2} \mathbf{S}^{\mathrm{T}} \mathbf{B} \mathbf{S} \mathbf{\Lambda}^{-1 / 2}\right| \\\\
& =|\mathbf{A}|\left|\mathbf{I}+\mathbf{\Lambda}^{-1 / 2} \mathbf{S}^{\mathrm{T}} \mathbf{B} \mathbf{S} \mathbf{\Lambda}^{-1 / 2}\right|
\end{aligned}
\end{equation}

If $\mathbf{B}=0$ then $|\mathbf{A}+\mathbf{B}|=|\mathbf{A}|$. If $\mathbf{B} \neq 0$, then the matrix $\mathbf{\Lambda}^{-1 / 2} \mathbf{S}^{\mathrm{T}} \mathbf{B} \mathbf{S} \mathbf{\Lambda}^{-1 / 2}$ will be positive semidefinite with at least one positive eigenvalue. Hence we have $\left|\mathbf{I}+\mathbf{\Lambda}^{-1 / 2} \mathbf{S}^{\mathrm{T}} \mathbf{B} \mathbf{S} \mathbf{\Lambda}^{-1 / 2}\right|>1$ and $|\mathbf{A}+\mathbf{B}|>|\mathbf{A}|$.
{{</proof>}}

{{<theorem "Simultaneous diagonalization" theorem_1_23>}}
Let $\mathbf{A}$ be positive definite and $\mathbf{B}$ symmetric of the same order. Then there exist a nonsingular matrix $\mathbf{P}$ and a diagonal matrix $\mathbf{\Lambda}$ such that

\begin{equation}
\mathbf{A}=\mathbf{P} \mathbf{P}^{\mathrm{T}}, \quad \mathbf{B}=\mathbf{P} \mathbf{\Lambda} \mathbf{P}^{\mathrm{T}}
\end{equation}

If $\mathbf{B}$ is positive semidefinite, then so is $\mathbf{\Lambda}$.
{{</theorem>}}

{{<proof>}}
Let $\mathbf{C}=\mathbf{A}^{-1 / 2} \mathbf{B} \mathbf{A}^{-1 / 2}$. Since $\mathbf{C}$ is symmetric, there exist an orthogonal matrix $\mathbf{S}$ and a diagonal matrix $\mathbf{\Lambda}$ such that

\begin{equation}
\mathbf{S}^{\mathrm{T}} \mathbf{C} \mathbf{S}=\mathbf{\Lambda}, \quad \mathbf{S}^{\mathrm{T}} \mathbf{S}=\mathbf{I}
\end{equation}

Now define $\mathbf{P}=\mathbf{A}^{1 / 2} \mathbf{S}$. Then,

\begin{equation}
\mathbf{P} \mathbf{P}^{\mathrm{T}}=\mathbf{A}^{1 / 2} \mathbf{S} \mathbf{S}^{\mathrm{T}} \mathbf{A}^{1 / 2}=\mathbf{A}^{1 / 2} \mathbf{A}^{1 / 2}=\mathbf{A}
\end{equation}

and

\begin{equation}
\mathbf{P} \mathbf{\Lambda} \mathbf{P}^{\mathrm{T}}=\mathbf{A}^{1 / 2} \mathbf{S} \mathbf{\Lambda} \mathbf{S}^{\mathrm{T}} \mathbf{A}^{1 / 2}=\mathbf{A}^{1 / 2} \mathbf{C} \mathbf{A}^{1 / 2}=\mathbf{A}^{1 / 2} \mathbf{A}^{-1 / 2} \mathbf{B} \mathbf{A}^{-1 / 2} \mathbf{A}^{1 / 2}=\mathbf{B} .
\end{equation}

If $\mathbf{B}$ is positive semidefinite, then so is $\mathbf{C}$ and so is $\mathbf{\Lambda}$.
{{</proof>}}

For two symmetric matrices $\mathbf{A}$ and $\mathbf{B}$, we shall write $\mathbf{A} \geq \mathbf{B}$ (or $\mathbf{B} \leq \mathbf{A}$ ) if $\mathbf{A}-\mathbf{B}$ is positive semidefinite, and $\mathbf{A}>\mathbf{B}$ (or $\mathbf{B}<\mathbf{A}$ ) if $\mathbf{A}-\mathbf{B}$ is positive definite.

{{<theorem "Inverse order for positive definite matrices" theorem_1_24>}}
Let $\mathbf{A}$ and $\mathbf{B}$ be positive definite $n \times n$ matrices. Then $\mathbf{A}>\mathbf{B}$ if and only if $\mathbf{B}^{-1}>\mathbf{A}^{-1}$.
{{</theorem>}}

{{<proof>}}
By {{<theoremref theorem_1_23>}}, there exist a nonsingular matrix $\mathbf{P}$ and a positive definite diagonal matrix $\mathbf{\Lambda}=\operatorname{diag}\left(\lambda_{1}, \ldots, \lambda_{n}\right)$ such that

\begin{equation}
\mathbf{A}=\mathbf{P} \mathbf{P}^{\mathrm{T}}, \quad \mathbf{B}=\mathbf{P} \mathbf{\Lambda} \mathbf{P}^{\mathrm{T}}
\end{equation}

Then,

\begin{equation}
\mathbf{A}-\mathbf{B}=\mathbf{P}(\mathbf{I}-\mathbf{\Lambda}) \mathbf{P}^{\mathrm{T}}, \quad \mathbf{B}^{-1}-\mathbf{A}^{-1}=\mathbf{P}^{\mathrm{T}-1}\left(\mathbf{\Lambda}^{-1}-\mathbf{I}\right) \mathbf{P}^{-1} .
\end{equation}

If $\mathbf{A}-\mathbf{B}$ is positive definite, then $\mathbf{I}-\mathbf{\Lambda}$ is positive definite and hence $0<\lambda_{i}<$ $1(i=1, \ldots, n)$. This implies that $\mathbf{\Lambda}^{-1}-\mathbf{I}$ is positive definite and hence that $\mathbf{B}^{-1}-\mathbf{A}^{-1}$ is positive definite.
{{</proof>}}

{{<theorem "Determinant monotonicity" theorem_1_25>}}
Let $\mathbf{A}$ and $\mathbf{B}$ be positive definite matrices such that $\mathbf{A}-\mathbf{B}$ is positive semidefinite. Then, $|\mathbf{A}| \geq|\mathbf{B}|$ with equality if and only if $\mathbf{A}=\mathbf{B}$.
{{</theorem>}}

{{<proof>}}
Let $\mathbf{C}=\mathbf{A}-\mathbf{B}$. Then $\mathbf{A}=\mathbf{B}+\mathbf{C}$, where $\mathbf{B}$ is positive definite and $\mathbf{C}$ is positive semidefinite. Thus, by {{<theoremref theorem_1_22>}}, $|\mathbf{B}+\mathbf{C}| \geq|\mathbf{B}|$ with equality if and only if $\mathbf{C}=0$, that is, $|\mathbf{A}| \geq|\mathbf{B}|$ with equality if and only if $\mathbf{A}=\mathbf{B}$.
{{</proof>}}

A useful special case of {{<theoremref theorem_1_25>}} is {{<theoremref theorem_1_26>}}.

{{<theorem "Identity characterization" theorem_1_26>}}
Let $\mathbf{A}$ be positive definite with $|\mathbf{A}|=1$. If $\mathbf{I}-\mathbf{A}$ is also positive semidefinite, then $\mathbf{A}=\mathbf{I}$.
{{</theorem>}}

{{<proof>}}
This follows immediately from {{<theoremref theorem_1_25>}}.
{{</proof>}}

## Three further results for positive definite matrices

Let us now prove {{<theoremref theorem_1_27>}}.

{{<theorem "Block matrix determinant and positive definiteness" theorem_1_27>}}
Let $\mathbf{A}$ be a positive definite $n \times n$ matrix, and let $\mathbf{B}$ be the $(n+1) \times(n+1)$ matrix

\begin{equation}
\mathbf{B}=\left(\begin{array}{ll}
\mathbf{A} & \mathbf{b} \\\\
\mathbf{b}^{\mathrm{T}} & \alpha
\end{array}\right)
\end{equation}

Then,
(i) $|\mathbf{B}| \leq \alpha|\mathbf{A}|$ with equality if and only if $\mathbf{b}=0$; and
(ii) $\mathbf{B}$ is positive definite if and only if $|\mathbf{B}|>0$.
{{</theorem>}}

{{<proof>}}
Define the $(n+1) \times(n+1)$ matrix

\begin{equation}
\mathbf{P}=\left(\begin{array}{cc}
\mathbf{I}_{n} & -\mathbf{A}^{-1} \mathbf{b} \\\\
\mathbf{0}^{\mathrm{T}} & 1
\end{array}\right)
\end{equation}

Then,

\begin{equation}
\mathbf{P}^{\mathrm{T}} \mathbf{B} \mathbf{P}=\left(\begin{array}{cc}
\mathbf{A} & 0 \\\\
\mathbf{0}^{\mathrm{T}} & \alpha-\mathbf{b}^{\mathrm{T}} \mathbf{A}^{-1} \mathbf{b}
\end{array}\right)
\end{equation}

so that

\begin{equation}
|\mathbf{B}|=\left|\mathbf{P}^{\mathrm{T}} \mathbf{B} \mathbf{P}\right|=|\mathbf{A}|\left(\alpha-\mathbf{b}^{\mathrm{T}} \mathbf{A}^{-1} \mathbf{b}\right) .
\label{eq:det_block}
\end{equation}

(Compare Exercise 2 in Section 1.11.) Then (i) is an immediate consequence of \eqref{eq:det_block}. To prove (ii) we note that $|\mathbf{B}|>0$ if and only if $\alpha-\mathbf{b}^{\mathrm{T}} \mathbf{A}^{-1} \mathbf{b}>0$ (from \eqref{eq:det_block}), which is the case if and only if $\mathbf{P}^{\mathrm{T}} \mathbf{B} \mathbf{P}$ is positive definite (from the previous equation). This in turn is true if and only if $\mathbf{B}$ is positive definite.
{{</proof>}}

An immediate consequence of {{<theoremref theorem_1_27>}}, proved by induction, is the following.

{{<theorem "Hadamard's inequality" theorem_1_28>}}
If $\mathbf{A}=\left(a_{ij}\right)$ is a positive definite $n \times n$ matrix, then

\begin{equation}
|\mathbf{A}| \leq \prod_{i=1}^{n} a_{ii}
\end{equation}

with equality if and only if $\mathbf{A}$ is diagonal.
{{</theorem>}}

Another consequence of {{<theoremref theorem_1_27>}} is {{<theoremref theorem_1_29>}}.

{{<theorem "Principal minor test" theorem_1_29>}}
A symmetric $n \times n$ matrix $\mathbf{A}$ is positive definite if and only if all principal minors $\left|\mathbf{A}_{k}\right|(k=1, \ldots, n)$ are positive.
{{</theorem>}}

Note. The $k \times k$ matrix $\mathbf{A}\_{k}$ is obtained from $\mathbf{A}$ by deleting the last $n-k$ rows and columns of $\mathbf{A}$. Notice that $\mathbf{A}\_{n}=\mathbf{A}$.

{{<proof>}}
Let $\mathbf{E}\_{k}=\left(\mathbf{I}\_{k}: 0\right)$ be a $k \times n$ matrix, so that $\mathbf{A}\_{k}=\mathbf{E}\_{k} \mathbf{A} \mathbf{E}\_{k}^{\mathrm{T}}$. Let $\mathbf{y}$ be an arbitrary $k \times 1$ vector, $\mathbf{y} \neq 0$. Then,

\begin{equation}
\mathbf{y}^{\mathrm{T}} \mathbf{A}\_{k} \mathbf{y}=\left(\mathbf{E}\_{k}^{\mathrm{T}} \mathbf{y}\right)^{\mathrm{T}} \mathbf{A}\left(\mathbf{E}\_{k}^{\mathrm{T}} \mathbf{y}\right)>0
\end{equation}

since $\mathbf{E}\_{k}^{\mathrm{T}} \mathbf{y} \neq 0$ and $\mathbf{A}$ is positive definite. Hence, $\mathbf{A}\_{k}$ is positive definite and, in particular, $\left|\mathbf{A}\_{k}\right|>0$. The converse follows by repeated application of {{<theoremref theorem_1_27>}}(ii).
{{</proof>}}

**Exercises**

1. If $\mathbf{A}$ is positive definite show that the matrix

\begin{equation}
\left(\begin{array}{cc}
\mathbf{A} & \mathbf{b} \\\\
\mathbf{b}^{\mathrm{T}} & \mathbf{b}^{\mathrm{T}} \mathbf{A}^{-1} \mathbf{b}
\end{array}\right)
\end{equation}

is positive semidefinite and singular, and find the eigenvector associated with the zero eigenvalue.

2. Hence show that, for positive definite $\mathbf{A}$,

\begin{equation}
\mathbf{x}^{\mathrm{T}} \mathbf{A} \mathbf{x}-2 \mathbf{b}^{\mathrm{T}} \mathbf{x} \geq-\mathbf{b}^{\mathrm{T}} \mathbf{A}^{-1} \mathbf{b}
\end{equation}

for every $\mathbf{x}$, with equality if and only if $\mathbf{x}=\mathbf{A}^{-1} \mathbf{b}$.

## A useful result

If $\mathbf{A}$ is a positive definite $n \times n$ matrix, then, in accordance with {{<theoremref theorem_1_28>}},

\begin{equation}
|\mathbf{A}|=\prod_{i=1}^{n} a_{ii}
\label{eq:diagonal_det}
\end{equation}

if and only if $\mathbf{A}$ is diagonal. If $\mathbf{A}$ is merely symmetric, then \eqref{eq:diagonal_det}, while obviously necessary, is no longer sufficient for the diagonality of $\mathbf{A}$. For example, the matrix

\begin{equation}
\mathbf{A}=\left(\begin{array}{lll}
2 & 3 & 3 \\\\
3 & 2 & 3 \\\\
3 & 3 & 2
\end{array}\right)
\end{equation}

has determinant $|\mathbf{A}|=8$ (its eigenvalues are $-1,-1$, and 8 ), thus satisfying \eqref{eq:diagonal_det}, but $\mathbf{A}$ is not diagonal.

{{<theoremref theorem_1_30>}} gives a necessary and sufficient condition for the diagonality of a symmetric matrix.

{{<theorem "Diagonal matrix characterization" theorem_1_30>}}
A symmetric matrix is diagonal if and only if its eigenvalues and its diagonal elements coincide.
{{</theorem>}}

{{<proof>}}
Let $\mathbf{A}=\left(a_{ij}\right)$ be a symmetric $n \times n$ matrix. The 'only if' part of the theorem is trivial. To prove the 'if' part, assume that $\lambda_{i}(\mathbf{A})=a_{ii}, i=1, \ldots, n$, and consider the matrix

\begin{equation}
\mathbf{B}=\mathbf{A}+k \mathbf{I},
\end{equation}

where $k>0$ is such that $\mathbf{B}$ is positive definite. Then,

\begin{equation}
\lambda_{i}(\mathbf{B})=\lambda_{i}(\mathbf{A})+k=a_{ii}+k=b_{ii} \quad(i=1, \ldots, n),
\end{equation}

and hence

\begin{equation}
|\mathbf{B}|=\prod_{1}^{n} \lambda_{i}(\mathbf{B})=\prod_{i=1}^{n} b_{ii} .
\end{equation}

It then follows from {{<theoremref theorem_1_28>}} that $\mathbf{B}$ is diagonal, and hence that $\mathbf{A}$ is diagonal.
{{</proof>}}

## Symmetric matrix functions

Let $\mathbf{A}$ be a square matrix of order $n \times n$. The $\operatorname{trace} \operatorname{tr} \mathbf{A}$ and the determinant $|\mathbf{A}|$ are examples of scalar functions of $\mathbf{A}$. We can also consider matrix functions, for example, the inverse $\mathbf{A}^{-1}$. The general definition of a matrix function is somewhat complicated, but for symmetric matrices it is easy. So, let us assume that $\mathbf{A}$ is symmetric.

We known from that any symmetric $n \times n$ matrix $\mathbf{A}$ can be diagonalized, which means that there exists an orthogonal matrix $\mathbf{S}$ and a diagonal matrix $\mathbf{\Lambda}$ (containing the eigenvalues of $\mathbf{A}$ ) such that $\mathbf{S}^{\mathrm{T}} \mathbf{A} \mathbf{S}=\mathbf{\Lambda}$. Let $\lambda_{i}$ denote the $i$ th diagonal element of $\mathbf{\Lambda}$ and let $\phi$ be a function so that $\phi(\lambda)$ is defined, for example, $\phi(\lambda)=\sqrt{\lambda}$ or $1 / \lambda$ or $\log \lambda$ or $e^{\lambda}$.

We now define the matrix function $F$ as

\begin{equation}
F(\mathbf{\Lambda})=\left(\begin{array}{cccc}
\phi\left(\lambda_{1}\right) & 0 & \ldots & 0 \\\\
0 & \phi\left(\lambda_{2}\right) & \ldots & 0 \\\\
\vdots & \vdots & & \vdots \\\\
0 & 0 & \ldots & \phi\left(\lambda_{n}\right)
\end{array}\right),
\end{equation}

and then

\begin{equation}
F(\mathbf{A})=\mathbf{S} F(\mathbf{\Lambda}) \mathbf{S}^{\mathrm{T}}
\end{equation}

For example, if $\mathbf{A}$ is nonsingular then all $\lambda_{i}$ are nonzero, and letting $\phi(\lambda)=$ $1 / \lambda$, we have

\begin{equation}
F(\mathbf{\Lambda})=\mathbf{\Lambda}^{-1}=\left(\begin{array}{cccc}
1 / \lambda_{1} & 0 & \ldots & 0 \\\\
0 & 1 / \lambda_{2} & \ldots & 0 \\\\
\vdots & \vdots & & \vdots \\\\
0 & 0 & \ldots & 1 / \lambda_{n}
\end{array}\right)
\end{equation}

and hence $\mathbf{A}^{-1}=\mathbf{S} \mathbf{\Lambda}^{-1} \mathbf{S}^{\mathrm{T}}$. To check, we have

\begin{equation}
\mathbf{A} \mathbf{A}^{-1}=\mathbf{S} \mathbf{\Lambda} \mathbf{S}^{\mathrm{T}} \mathbf{S} \mathbf{\Lambda}^{-1} \mathbf{S}^{\mathrm{T}}=\mathbf{S} \mathbf{\Lambda} \mathbf{\Lambda}^{-1} \mathbf{S}^{\mathrm{T}}=\mathbf{S} \mathbf{S}^{\mathrm{T}}=\mathbf{I}_{n}
\end{equation}

as we should.
Similarly, if $\mathbf{A}$ is positive semidefinite, then $\mathbf{A}^{1 / 2}=\mathbf{S} \mathbf{\Lambda}^{1 / 2} \mathbf{S}^{\mathrm{T}}$ and

$$
\mathbf{A}^{1 / 2} \mathbf{A}^{1 / 2}=\mathbf{S} \boldsymbol{\Lambda}^{1 / 2} \mathbf{S}^{\mathrm{T}} \mathbf{S} \boldsymbol{\Lambda}^{1 / 2} \mathbf{S}^{\mathrm{T}}=\mathbf{S} \boldsymbol{\Lambda}^{1 / 2} \boldsymbol{\Lambda}^{1 / 2} \mathbf{S}^{\mathrm{T}}=\mathbf{S} \boldsymbol{\Lambda} \mathbf{S}^{\mathrm{T}}=\mathbf{A},
$$

again as it should be. Also, when $\mathbf{A}$ is positive definite (hence nonsingular), then $\mathbf{A}^{-1 / 2}=\mathbf{S} \boldsymbol{\Lambda}^{-1 / 2} \mathbf{S}^{\mathrm{T}}$ and

$$
\begin{aligned}
\left(\mathbf{A}^{1 / 2}\right)^{-1} & =\left(\mathbf{S} \boldsymbol{\Lambda}^{1 / 2} \mathbf{S}^{\mathrm{T}}\right)^{-1}=\mathbf{S}\left(\boldsymbol{\Lambda}^{1 / 2}\right)^{-1} \mathbf{S}^{\mathrm{T}}=\mathbf{S}\left(\boldsymbol{\Lambda}^{-1}\right)^{1 / 2} \mathbf{S}^{\mathrm{T}} \\\\
& =\left(\mathbf{S} \boldsymbol{\Lambda}^{-1} \mathbf{S}^{\mathrm{T}}\right)^{1 / 2}=\left(\mathbf{A}^{-1}\right)^{1 / 2}
\end{aligned}
$$

so that this expression is unambiguously defined.
Symmetric matrix functions are thus always defined through their eigenvalues. For example, the logarithm or exponential of $\mathbf{A}$ is not the matrix with elements $\log a\_{i j}$ or $e^{a\_{i j}}$, but rather a matrix whose eigenvalues are $\log \lambda\_{i}$ or $e^{\lambda\_{i}}$ and whose eigenvectors are the same as the eigenvectors of $\mathbf{A}$. This is similar to the definition of a positive definite matrix, which is not a matrix all whose elements are positive, but rather a matrix all whose eigenvalues are positive.

## Miscellaneous exercises

**Exercises**

1. If $\mathbf{A}$ and $\mathbf{B}$ are square matrices such that $\mathbf{A}\mathbf{B}=0, \mathbf{A} \neq 0, \mathbf{B} \neq 0$, then prove that $|\mathbf{A}|=|\mathbf{B}|=0$.
2. If $\mathbf{x}$ and $\mathbf{y}$ are vectors of the same order, prove that $\mathbf{x}^{\mathrm{T}} \mathbf{y}=\operatorname{tr} \mathbf{y} \mathbf{x}^{\mathrm{T}}$.
3. Let

$$
\mathbf{A}=\left(\begin{array}{ll}
\mathbf{A}\_{11} & \mathbf{A}\_{12} \\\\
\mathbf{A}\_{21} & \mathbf{A}\_{22}
\end{array}\right)
$$

Show that

$$
|\mathbf{A}|=\left|\mathbf{A}\_{11}\right|\left|\mathbf{A}\_{22}-\mathbf{A}\_{21} \mathbf{A}\_{11}^{-1} \mathbf{A}\_{12}\right|
$$

if $\mathbf{A}\_{11}$ is nonsingular, and

$$
|\mathbf{A}|=\left|\mathbf{A}\_{22}\right|\left|\mathbf{A}\_{11}-\mathbf{A}\_{12} \mathbf{A}\_{22}^{-1} \mathbf{A}\_{21}\right|
$$

if $\mathbf{A}\_{22}$ is nonsingular.
4. Show that $(\mathbf{I}-\mathbf{A}\mathbf{B})^{-1}=\mathbf{I}+\mathbf{A}(\mathbf{I}-\mathbf{B}\mathbf{A})^{-1}\mathbf{B}$, if the inverses exist.
5. Show that

$$
(\alpha \mathbf{I}-\mathbf{A})^{-1}-(\beta \mathbf{I}-\mathbf{A})^{-1}=(\beta-\alpha)(\beta \mathbf{I}-\mathbf{A})^{-1}(\alpha \mathbf{I}-\mathbf{A})^{-1} .
$$

6. If $\mathbf{A}$ is positive definite, show that $\mathbf{A}+\mathbf{A}^{-1}-2 \mathbf{I}$ is positive semidefinite.
7. For any symmetric matrices $\mathbf{A}$ and $\mathbf{B}$, show that $\mathbf{A}\mathbf{B}-\mathbf{B}\mathbf{A}$ is skewsymmetric.
8. Let $\mathbf{A}$ and $\mathbf{B}$ be two $m \times n$ matrices of rank $r$. If $\mathbf{A}\mathbf{A}^{\mathrm{T}}=\mathbf{B}\mathbf{B}^{\mathrm{T}}$ then $\mathbf{A}=\mathbf{B}\mathbf{Q}$, where $\mathbf{Q}\mathbf{Q}^{\mathrm{T}}$ (and hence $\mathbf{Q}^{\mathrm{T}} \mathbf{Q}$ ) is idempotent of rank $k \geq r$ (Neudecker and van de Velden 2000).
9. Let $\mathbf{A}$ be an $m \times n$ matrix partitioned as $\mathbf{A}=\left(\mathbf{A}\_{1}: \mathbf{A}\_{2}\right)$ and satisfying $\mathbf{A}\_{1}^{\mathrm{T}} \mathbf{A}\_{2}=0$ and $r\left(\mathbf{A}\_{1}\right)+r\left(\mathbf{A}\_{2}\right)=m$. Then, for any positive semidefinite matrix $\mathbf{V}$, we have

$$
r(\mathbf{V})=r\left(\mathbf{A}\_{1}\right)+r\left(\mathbf{A}\_{2}^{\mathrm{T}} \mathbf{V} \mathbf{A}\_{2}\right) \Longleftrightarrow r(\mathbf{V})=r\left(\mathbf{V}: \mathbf{A}\_{1}\right)
$$

10. Prove that the eigenvalues $\lambda\_{i}$ of $(\mathbf{A}+\mathbf{B})^{-1} \mathbf{A}$, where $\mathbf{A}$ is positive semidefinite and $\mathbf{B}$ is positive definite, satisfy $0 \leq \lambda\_{i}<1$.
11. Let $\mathbf{x}$ and $\mathbf{y}$ be $n \times 1$ vectors. Prove that $\mathbf{x} \mathbf{y}^{\mathrm{T}}$ has $n-1$ zero eigenvalues and one eigenvalue $\mathbf{x}^{\mathrm{T}} \mathbf{y}$.
12. Show that $\left|\mathbf{I}+\mathbf{x} \mathbf{y}^{\mathrm{T}}\right|=1+\mathbf{x}^{\mathrm{T}} \mathbf{y}$.
13. Let $\mu=1+\mathbf{x}^{\mathrm{T}} \mathbf{y}$. If $\mu \neq 0$, show that $\left(\mathbf{I}+\mathbf{x} \mathbf{y}^{\mathrm{T}}\right)^{-1}=\mathbf{I}-(1 / \mu) \mathbf{x} \mathbf{y}^{\mathrm{T}}$.
14. Show that $\left(\mathbf{I}+\mathbf{A} \mathbf{A}^{\mathrm{T}}\right)^{-1} \mathbf{A}=\mathbf{A}\left(\mathbf{I}+\mathbf{A}^{\mathrm{T}} \mathbf{A}\right)^{-1}$.
15. Show that $\mathbf{A}\left(\mathbf{A}^{\mathrm{T}} \mathbf{A}\right)^{1 / 2}=\left(\mathbf{A} \mathbf{A}^{\mathrm{T}}\right)^{1 / 2} \mathbf{A}$.
16. (Monotonicity of the entropic complexity.) Let $\mathbf{A}\_{n}$ be a positive definite $n \times n$ matrix and define

$$
\phi(n)=\frac{n}{2} \log \operatorname{tr}\left(\mathbf{A}\_{n} / n\right)-\frac{1}{2} \log \left|\mathbf{A}\_{n}\right| .
$$

Let $\mathbf{A}\_{n+1}$ be a positive definite $(n+1) \times(n+1)$ matrix such that

$$
\mathbf{A}\_{n+1}=\left(\begin{array}{cc}
\mathbf{A}\_{n} & \mathbf{a}\_{n} \\\\
\mathbf{a}\_{n}^{\mathrm{T}} & \alpha\_{n}
\end{array}\right)
$$

Then,

$$
\phi(n+1) \geq \phi(n)
$$

with equality if and only if

$$
\mathbf{a}\_{n}=0, \quad \alpha\_{n}=\operatorname{tr} \mathbf{A}\_{n} / n
$$

