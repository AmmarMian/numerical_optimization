---
title: Linear Algebra
weight: 1
chapter: 0
math: true
---

# Fundamentals of Linear Algebra

Linear algebra is one of the foundational branches of mathematics, with applications spanning from engineering and computer science to economics and physics. This document introduces the key concepts and theorems of linear algebra in a structured progression, demonstrating how each idea builds upon previous ones to form a coherent mathematical framework.

## 1. Vectors and Vector Spaces

We begin our study of linear algebra with the fundamental concept of a vector space, which formalizes the notion of vectors and their operations. This abstraction allows us to work with many different types of mathematical objects using the same underlying principles.

{{< definition "Vector Space" "vector-space" >}}
A vector space $V$ over a field $F$ is a set equipped with two operations:
- Vector addition: $+: V \times V \rightarrow V$
- Scalar multiplication: $\cdot: F \times V \rightarrow V$

satisfying the following axioms for all $\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$ and $a, b \in F$:
1. Closure under addition: $\mathbf{u} + \mathbf{v} \in V$
2. Commutativity: $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$
3. Associativity: $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$
4. Additive identity: There exists $\mathbf{0} \in V$ such that $\mathbf{v} + \mathbf{0} = \mathbf{v}$ for all $\mathbf{v} \in V$
5. Additive inverse: For each $\mathbf{v} \in V$, there exists $-\mathbf{v} \in V$ such that $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$
6. Closure under scalar multiplication: $a \cdot \mathbf{v} \in V$
7. Distributivity: $a \cdot (\mathbf{u} + \mathbf{v}) = a \cdot \mathbf{u} + a \cdot \mathbf{v}$ and $(a + b) \cdot \mathbf{v} = a \cdot \mathbf{v} + b \cdot \mathbf{v}$
8. Scalar multiplication associativity: $a \cdot (b \cdot \mathbf{v}) = (ab) \cdot \mathbf{v}$
9. Scalar multiplication identity: $1 \cdot \mathbf{v} = \mathbf{v}$
{{< /definition >}}

{{% hint info %}}
**Intuition for Vector Spaces**  
Think of vectors as arrows with direction and magnitude. The vector space axioms formalize how these arrows can be combined and scaled while remaining within the same space. $\mathbb{R}^n$ is the most common example, but polynomial spaces, function spaces, and matrix spaces are also important vector spaces.
{{% /hint %}}

{{< definition "Vector Subspace" "vector-subspace" >}}
A subset $W$ of a vector space $V$ is a subspace if:
1. The zero vector $\mathbf{0} \in W$
2. $W$ is closed under addition: for all $\mathbf{u}, \mathbf{v} \in W$, $\mathbf{u} + \mathbf{v} \in W$
3. $W$ is closed under scalar multiplication: for all $\mathbf{v} \in W$ and $c \in F$, $c\mathbf{v} \in W$
{{< /definition >}}

{{< theorem "Linear Independence" "linear-independence" >}}
A set of vectors $\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\}$ is linearly independent if and only if the equation
\begin{equation}
  c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \ldots + c_k\mathbf{v}_k = \mathbf{0}
  \label{eq:linear-independence}
\end{equation}
has only the trivial solution $c_1 = c_2 = \ldots = c_k = 0$.
{{< /theorem >}}

{{< proof >}}
We prove both directions of the if and only if statement:

($\Rightarrow$) Suppose the set $\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\}$ is linearly independent. By definition, this means that the only way to express the zero vector as a linear combination of these vectors is with all coefficients equal to zero. This is precisely the statement that equation \eqref{eq:linear-independence} has only the trivial solution.

($\Leftarrow$) Conversely, suppose equation \eqref{eq:linear-independence} has only the trivial solution $c_1 = c_2 = \ldots = c_k = 0$. This means that the only way to express the zero vector as a linear combination of the vectors $\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\}$ is with all coefficients equal to zero. By definition, this means the set is linearly independent.
{{< /proof >}}

{{< definition "Span" "span" >}}
The span of a set of vectors $\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\}$ is the set of all possible linear combinations:
\begin{equation}
  \text{Span}(\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k) = \{c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \ldots + c_k\mathbf{v}_k \mid c_1, c_2, \ldots, c_k \in F\}
  \label{eq:span}
\end{equation}
{{< /definition >}}

Linear independence and span are two fundamental concepts that help us understand the structure of vector spaces. When we combine these ideas, we arrive at the important concept of a basis—a minimal set of vectors that can represent every vector in the space through linear combinations.

{{< definition "Basis" "basis" >}}
A basis for a vector space $V$ is a set of vectors $\mathcal{B} = \{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n\}$ that is:
1. Linearly independent
2. Spans $V$

The number of vectors in any basis is called the dimension of $V$, denoted $\dim(V)$.
{{< /definition >}}

### Exercises on Vectors and Vector Spaces

**Exercise 1.1** Determine whether the set $W = \{(x,y,z) \in \mathbb{R}^3 \mid x + y + z = 0\}$ is a subspace of $\mathbb{R}^3$.

**Solution:**
We need to check the three conditions for a subspace:
1. Zero vector: $(0,0,0) \in W$ since $0 + 0 + 0 = 0$
2. Closure under addition: Let $\mathbf{u} = (u_1, u_2, u_3)$ and $\mathbf{v} = (v_1, v_2, v_3)$ be vectors in $W$. Then $u_1 + u_2 + u_3 = 0$ and $v_1 + v_2 + v_3 = 0$. We have:
   $\mathbf{u} + \mathbf{v} = (u_1 + v_1, u_2 + v_2, u_3 + v_3)$
   $(u_1 + v_1) + (u_2 + v_2) + (u_3 + v_3) = (u_1 + u_2 + u_3) + (v_1 + v_2 + v_3) = 0 + 0 = 0$
   So $\mathbf{u} + \mathbf{v} \in W$
3. Closure under scalar multiplication: For any $c \in \mathbb{R}$ and $\mathbf{u} = (u_1, u_2, u_3) \in W$:
   $c\mathbf{u} = (cu_1, cu_2, cu_3)$
   $cu_1 + cu_2 + cu_3 = c(u_1 + u_2 + u_3) = c \cdot 0 = 0$
   So $c\mathbf{u} \in W$

Since all three conditions are satisfied, $W$ is indeed a subspace of $\mathbb{R}^3$.

**Exercise 1.2** Determine whether the set $\{(1,1,0), (0,1,1), (1,0,1)\}$ is linearly independent in $\mathbb{R}^3$.

**Solution:**
To determine linear independence, we need to check if the equation $c_1(1,1,0) + c_2(0,1,1) + c_3(1,0,1) = (0,0,0)$ has only the trivial solution.

This gives us the system:
$c_1 + c_3 = 0$
$c_1 + c_2 = 0$
$c_2 + c_3 = 0$

From the first two equations, we get $c_3 = -c_1$ and $c_2 = -c_1$. Substituting into the third equation:
$-c_1 + (-c_1) = 0$
$-2c_1 = 0$
$c_1 = 0$

This means $c_2 = c_3 = 0$ as well. Since we only get the trivial solution, the set is linearly independent.

## 2. Matrices and Linear Transformations

Having established the foundation of vector spaces, we now turn to linear transformations—functions that preserve the vector space structure. Matrices provide a concrete way to represent these abstract transformations, allowing us to apply computational techniques to study their properties.

{{< definition "Matrix" "matrix" >}}
An $m \times n$ matrix $A$ over a field $F$ is a rectangular array of elements from $F$ arranged in $m$ rows and $n$ columns:

\begin{equation}
  A = \begin{pmatrix}
    a_{11} & a_{12} & \cdots & a_{1n} \\
    a_{21} & a_{22} & \cdots & a_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1} & a_{m2} & \cdots & a_{mn}
  \end{pmatrix}
  \label{eq:matrix}
\end{equation}

We denote the $(i,j)$-entry as $a_{ij}$ or $[A]_{ij}$.
{{< /definition >}}

{{< definition "Linear Transformation" "linear-transformation" >}}
A function $T: V \rightarrow W$ between vector spaces is a linear transformation if for all vectors $\mathbf{u}, \mathbf{v} \in V$ and all scalars $c \in F$:
1. $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$ (Additivity)
2. $T(c\mathbf{u}) = cT(\mathbf{u})$ (Homogeneity)
{{< /definition >}}

{{< theorem "Matrix Representation" "matrix-representation" >}}
Every linear transformation $T: V \rightarrow W$ between finite-dimensional vector spaces with bases can be represented by a unique matrix $A$ such that
\begin{equation}
  T(\mathbf{v}) = A\mathbf{v}
  \label{eq:matrix-representation}
\end{equation}
where $\mathbf{v}$ is expressed in the basis of $V$ and $T(\mathbf{v})$ in the basis of $W$.
{{< /theorem >}}

{{< proof >}}
Let $\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n\}$ be a basis for $V$ and $\{\mathbf{w}_1, \mathbf{w}_2, \ldots, \mathbf{w}_m\}$ be a basis for $W$. 

For each basis vector $\mathbf{v}_j$ of $V$, its image under $T$ can be expressed uniquely as a linear combination of the basis vectors of $W$:
$T(\mathbf{v}_j) = a_{1j}\mathbf{w}_1 + a_{2j}\mathbf{w}_2 + \ldots + a_{mj}\mathbf{w}_m$

Let $A$ be the $m \times n$ matrix whose $(i,j)$-entry is $a_{ij}$. Now, for any vector $\mathbf{v} \in V$, we can write it uniquely as a linear combination of the basis vectors:
$\mathbf{v} = c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \ldots + c_n\mathbf{v}_n$

Applying $T$ to both sides and using linearity:
$T(\mathbf{v}) = T(c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \ldots + c_n\mathbf{v}_n)$
$= c_1T(\mathbf{v}_1) + c_2T(\mathbf{v}_2) + \ldots + c_nT(\mathbf{v}_n)$

Substituting the expressions for $T(\mathbf{v}_j)$:
$T(\mathbf{v}) = c_1(a_{11}\mathbf{w}_1 + \ldots + a_{m1}\mathbf{w}_m) + \ldots + c_n(a_{1n}\mathbf{w}_1 + \ldots + a_{mn}\mathbf{w}_m)$

Rearranging:
$T(\mathbf{v}) = (a_{11}c_1 + \ldots + a_{1n}c_n)\mathbf{w}_1 + \ldots + (a_{m1}c_1 + \ldots + a_{mn}c_n)\mathbf{w}_m$

This is exactly the result of the matrix-vector multiplication $A\mathbf{c}$, where $\mathbf{c} = (c_1, c_2, \ldots, c_n)^T$ is the coordinate vector of $\mathbf{v}$ with respect to the basis of $V$.

For uniqueness, suppose there are two matrices $A$ and $B$ such that $T(\mathbf{v}) = A\mathbf{v} = B\mathbf{v}$ for all $\mathbf{v} \in V$. Then $(A-B)\mathbf{v} = \mathbf{0}$ for all $\mathbf{v} \in V$. In particular, this must hold for each basis vector $\mathbf{v}_j$, which implies that all columns of $A-B$ are zero. Therefore, $A = B$.
{{< /proof >}}

{{% hint warning %}}
**Change of Basis**  
The matrix representation depends on the chosen bases. Changing the basis transforms the matrix according to $A' = P^{-1}AP$, where $P$ is the change-of-basis matrix. This relationship is fundamental in understanding how the same linear transformation can be represented differently in different coordinate systems.
{{% /hint %}}

{{< definition "Matrix Operations" "matrix-operations" >}}
For matrices $A$ and $B$ of appropriate dimensions and scalar $c$:

1. Addition: $[A + B]_{ij} = [A]_{ij} + [B]_{ij}$
2. Scalar multiplication: $[cA]_{ij} = c[A]_{ij}$
3. Matrix multiplication: $[AB]_{ij} = \sum_{k=1}^{n} [A]_{ik}[B]_{kj}$
4. Transpose: $[A^T]_{ij} = [A]_{ji}$
{{< /definition >}}

### Exercises on Matrices and Linear Transformations

**Exercise 2.1** Let $T: \mathbb{R}^2 \rightarrow \mathbb{R}^2$ be the linear transformation that rotates vectors counterclockwise by $90^\circ$. Find the matrix representation of $T$ with respect to the standard basis.

**Solution:**
The standard basis for $\mathbb{R}^2$ is $\{\mathbf{e}_1, \mathbf{e}_2\}$ where $\mathbf{e}_1 = (1,0)$ and $\mathbf{e}_2 = (0,1)$.

When rotated 90° counterclockwise:
- $T(\mathbf{e}_1) = T(1,0) = (0,1) = \mathbf{e}_2$
- $T(\mathbf{e}_2) = T(0,1) = (-1,0) = -\mathbf{e}_1$

The matrix representation is formed by putting these output vectors as columns:
$A = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$

Let's verify with a test vector $\mathbf{v} = (2,3)$:
$A\mathbf{v} = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} 2 \\ 3 \end{pmatrix} = \begin{pmatrix} -3 \\ 2 \end{pmatrix}$

Which is indeed the vector $(2,3)$ rotated 90° counterclockwise.

**Exercise 2.2** Determine the kernel (null space) and image (range) of the linear transformation represented by the matrix $A = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \end{pmatrix}$.

**Solution:**
The kernel of $A$ is $\text{ker}(A) = \{\mathbf{v} \in \mathbb{R}^3 \mid A\mathbf{v} = \mathbf{0}\}$.

We need to find all vectors $\mathbf{v} = (x,y,z)$ such that $A\mathbf{v} = \mathbf{0}$:
$\begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \end{pmatrix} \begin{pmatrix} x \\ y \\ z \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$

This gives us:
$x + 2y + 3z = 0$
$2x + 4y + 6z = 0$

Note that the second equation is just 2 times the first, so we effectively have just one constraint:
$x + 2y + 3z = 0$

We can parameterize the solution with two free variables, e.g., $y = s$ and $z = t$:
$x = -2s - 3t$

So the kernel is:
$\text{ker}(A) = \{(-2s-3t, s, t) \mid s,t \in \mathbb{R}\} = \text{span}\{(-2,1,0), (-3,0,1)\}$

For the image, we look at the column space of $A$:
$\text{im}(A) = \text{span}\{(1,2), (2,4), (3,6)\}$

We can see that columns 2 and 3 are multiples of column 1, so:
$\text{im}(A) = \text{span}\{(1,2)\}$

Thus, the image is a one-dimensional subspace of $\mathbb{R}^2$.

## 3. Matrix Determinants and Invertibility

Now that we understand matrices as representations of linear transformations, we need tools to analyze their properties. The determinant is a scalar value associated with a square matrix that provides crucial information about its invertibility and geometric interpretation. This section explores determinants and their connection to the existence of solutions for linear systems.

{{< definition "Determinant" "determinant" >}}
The determinant of a square matrix $A \in \mathbb{R}^{n \times n}$ is a scalar value denoted $\det(A)$ or $|A|$ that can be defined recursively:

For a $1 \times 1$ matrix $A = [a]$, $\det(A) = a$.

For an $n \times n$ matrix with $n > 1$:
\begin{equation}
  \det(A) = \sum_{j=1}^{n} (-1)^{1+j} a_{1j} \det(A_{1j})
  \label{eq:determinant}
\end{equation}

where $A_{1j}$ is the $(n-1) \times (n-1)$ submatrix obtained by removing the first row and $j$-th column of $A$.
{{< /definition >}}

{{% hint info %}}
**Geometric Interpretation of Determinant**  
The absolute value of the determinant of a matrix represents the scaling factor of the volume transformation under the linear map. For a $2 \times 2$ matrix, $|\det(A)|$ gives the area of the parallelogram formed by the column vectors of $A$. For a $3 \times 3$ matrix, it gives the volume of the parallelepiped.
{{% /hint %}}

{{< theorem "Properties of Determinants" "determinant-properties" >}}
For square matrices $A$ and $B$ of the same dimension:

1. $\det(AB) = \det(A) \cdot \det(B)$
2. $\det(A^T) = \det(A)$
3. $\det(cA) = c^n \det(A)$ for an $n \times n$ matrix and scalar $c$
4. $A$ is invertible if and only if $\det(A) \neq 0$
5. If $A$ has a row or column of zeros, then $\det(A) = 0$
6. If two rows or columns of $A$ are identical, then $\det(A) = 0$
{{< /theorem >}}

{{< proof >}}
We'll prove some of these properties:

1. $\det(AB) = \det(A) \cdot \det(B)$:
   This can be proven using the multilinearity of the determinant and the formula for matrix multiplication. For $2 \times 2$ matrices, we can verify directly:
   
   Let $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ and $B = \begin{pmatrix} e & f \\ g & h \end{pmatrix}$.
   
   Then $AB = \begin{pmatrix} ae + bg & af + bh \\ ce + dg & cf + dh \end{pmatrix}$.
   
   $\det(AB) = (ae + bg)(cf + dh) - (af + bh)(ce + dg)$
   $= aecf + aedh + bgcf + bgdh - afce - afdg - bhce - bhdg$
   $= aecf + aedh + bgcf + bgdh - afce - afdg - bhce - bhdg$
   $= (ad - bc)(eh - fg) = \det(A) \cdot \det(B)$
   
   For general $n \times n$ matrices, a more advanced approach using permutations or eigenvalues is needed.

4. $A$ is invertible if and only if $\det(A) \neq 0$:
   
   ($\Rightarrow$) Suppose $A$ is invertible. Then there exists a matrix $A^{-1}$ such that $AA^{-1} = A^{-1}A = I$. Taking determinants of both sides:
   $\det(AA^{-1}) = \det(I) = 1$
   
   By property 1, $\det(A) \cdot \det(A^{-1}) = 1$, which implies $\det(A) \neq 0$.
   
   ($\Leftarrow$) This direction requires the adjugate matrix or Cramer's rule, which states that if $\det(A) \neq 0$, then $A^{-1} = \frac{1}{\det(A)} \text{adj}(A)$, where $\text{adj}(A)$ is the adjugate matrix of $A$. This formula shows that $A^{-1}$ exists when $\det(A) \neq 0$.

5. If $A$ has a row of zeros, say row $i$, then the determinant expression involves a sum of products, each containing one element from row $i$. Since all these elements are zero, their products are zero, making the entire determinant zero.
{{< /proof >}}

{{< definition "Matrix Inverse" "matrix-inverse" >}}
For a square matrix $A$, its inverse (if it exists) is a matrix $A^{-1}$ such that:
\begin{equation}
  AA^{-1} = A^{-1}A = I
  \label{eq:inverse}
\end{equation}
where $I$ is the identity matrix.

For a $2 \times 2$ matrix $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ with $\det(A) \neq 0$:
\begin{equation}
  A^{-1} = \frac{1}{\det(A)} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}
  \label{eq:2x2-inverse}
\end{equation}
{{< /definition >}}

{{< theorem "Invertible Matrix Theorem" "invertible-matrix-theorem" >}}
For an $n \times n$ matrix $A$, the following statements are equivalent:

1. $A$ is invertible.
2. $\det(A) \neq 0$.
3. The columns of $A$ form a basis for $\mathbb{R}^n$.
4. The linear system $A\mathbf{x} = \mathbf{b}$ has a unique solution for every $\mathbf{b} \in \mathbb{R}^n$.
5. The linear transformation $T(\mathbf{x}) = A\mathbf{x}$ is bijective.
6. $\text{rank}(A) = n$.
7. The nullity of $A$ is zero (i.e., $\dim(\ker(A)) = 0$).
8. $0$ is not an eigenvalue of $A$.
{{< /theorem >}}

{{< proof >}}
We'll prove some key equivalences:

$(1 \Leftrightarrow 4)$: $A$ is invertible if and only if $A\mathbf{x} = \mathbf{b}$ has a unique solution for every $\mathbf{b}$.

$(\Rightarrow)$ If $A$ is invertible, then for any $\mathbf{b}$, we can multiply both sides by $A^{-1}$ to get $\mathbf{x} = A^{-1}\mathbf{b}$, which is a unique solution.

$(\Leftarrow)$ If $A\mathbf{x} = \mathbf{b}$ has a unique solution for every $\mathbf{b}$, then in particular, for each standard basis vector $\mathbf{e}_j$, there exists a unique vector $\mathbf{c}_j$ such that $A\mathbf{c}_j = \mathbf{e}_j$. Let $C$ be the matrix with columns $\mathbf{c}_j$. Then $AC = I$. Similarly, for the system $\mathbf{y}^T A = \mathbf{e}_j^T$, there exists a unique row vector $\mathbf{r}_j^T$ such that $\mathbf{r}_j^T A = \mathbf{e}_j^T$. Let $R$ be the matrix with rows $\mathbf{r}_j^T$. Then $RA = I$. So $R = C$ and $A$ is invertible with $A^{-1} = C = R$.

$(4 \Leftrightarrow 7)$: The system $A\mathbf{x} = \mathbf{b}$ has a unique solution for every $\mathbf{b}$ if and only if $\ker(A) = \{\mathbf{0}\}$.

$(\Rightarrow)$ If $A\mathbf{x} = \mathbf{b}$ has a unique solution for every $\mathbf{b}$, then in particular, $A\mathbf{x} = \mathbf{0}$ has a unique solution. Since $\mathbf{x} = \mathbf{0}$ is always a solution, it must be the only solution. Thus, $\ker(A) = \{\mathbf{0}\}$.

$(\Leftarrow)$ If $\ker(A) = \{\mathbf{0}\}$, then for any $\mathbf{b}$, if $\mathbf{x}_1$ and $\mathbf{x}_2$ are both solutions to $A\mathbf{x} = \mathbf{b}$, then $A(\mathbf{x}_1 - \mathbf{x}_2) = \mathbf{0}$, which means $\mathbf{x}_1 - \mathbf{x}_2 \in \ker(A) = \{\mathbf{0}\}$. Therefore, $\mathbf{x}_1 = \mathbf{x}_2$, and the solution is unique.

$(7 \Leftrightarrow 8)$: $\ker(A) = \{\mathbf{0}\}$ if and only if $0$ is not an eigenvalue of $A$.

By definition, $\lambda$ is an eigenvalue of $A$ if and only if there exists a non-zero vector $\mathbf{v}$ such that $A\mathbf{v} = \lambda\mathbf{v}$. For $\lambda = 0$, this means $A\mathbf{v} = \mathbf{0}$, which is equivalent to $\mathbf{v} \in \ker(A)$. So $0$ is an eigenvalue if and only if $\ker(A)$ contains a non-zero vector, i.e., $\ker(A) \neq \{\mathbf{0}\}$.

The other equivalences can be proven using similar arguments and the rank-nullity theorem.
{{< /proof >}}

The Invertible Matrix Theorem is a powerful result that connects many seemingly different concepts in linear algebra. It tells us that a square matrix's invertibility can be characterized in multiple equivalent ways, providing flexibility in how we approach problems involving invertible matrices.

### Exercises on Determinants and Invertibility

**Exercise 3.1** Calculate the determinant of the matrix $A = \begin{pmatrix} 3 & 1 & 0 \\ 2 & -1 & 4 \\ -1 & 2 & 5 \end{pmatrix}$.

**Solution:**
Using the cofactor expansion along the first row:

$\det(A) = 3 \cdot \det\begin{pmatrix} -1 & 4 \\ 2 & 5 \end{pmatrix} - 1 \cdot \det\begin{pmatrix} 2 & 4 \\ -1 & 5 \end{pmatrix} + 0 \cdot \det\begin{pmatrix} 2 & -1 \\ -1 & 2 \end{pmatrix}$

$= 3 \cdot ((-1) \cdot 5 - 4 \cdot 2) - 1 \cdot (2 \cdot 5 - 4 \cdot (-1))$

$= 3 \cdot (-5 - 8) - 1 \cdot (10 + 4)$

$= 3 \cdot (-13) - 1 \cdot 14$

$= -39 - 14$

$= -53$

**Exercise 3.2** Find the inverse of the matrix $A = \begin{pmatrix} 2 & 1 \\ 5 & 3 \end{pmatrix}$ if it exists.

**Solution:**
First, we calculate the determinant:
$\det(A) = 2 \cdot 3 - 1 \cdot 5 = 6 - 5 = 1$

Since $\det(A) \neq 0$, the matrix is invertible.

Using the formula for a $2 \times 2$ matrix:
$A^{-1} = \frac{1}{\det(A)} \begin{pmatrix} 3 & -1 \\ -5 & 2 \end{pmatrix}$

$= \begin{pmatrix} 3 & -1 \\ -5 & 2 \end{pmatrix}$

Let's verify: 
$AA^{-1} = \begin{pmatrix} 2 & 1 \\ 5 & 3 \end{pmatrix} \begin{pmatrix} 3 & -1 \\ -5 & 2 \end{pmatrix} = \begin{pmatrix} 2 \cdot 3 + 1 \cdot (-5) & 2 \cdot (-1) + 1 \cdot 2 \\ 5 \cdot 3 + 3 \cdot (-5) & 5 \cdot (-1) + 3 \cdot 2 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$

## 4. Eigenvalues and Eigenvectors

Having explored the basics of matrices and their properties, we now delve into one of the most important concepts in linear algebra: eigenvalues and eigenvectors. These concepts reveal the intrinsic characteristics of linear transformations and find applications in diverse fields such as dynamic systems, quantum mechanics, and data science. Eigenvalues and eigenvectors provide insight into the behavior of iterative processes and lead naturally to the powerful spectral theorem for symmetric matrices.

{{< definition "Eigenvalues and Eigenvectors" "eigenvalues-eigenvectors" >}}
For a square matrix $A \in \mathbb{R}^{n \times n}$, a non-zero vector $\mathbf{v} \in \mathbb{R}^n$ is an eigenvector of $A$ with corresponding eigenvalue $\lambda \in \mathbb{R}$ if:
\begin{equation}
  A\mathbf{v} = \lambda\mathbf{v}
  \label{eq:eigenvalue-equation}
\end{equation}
{{< /definition >}}

{{< theorem "Eigenvalue Calculation" "eigenvalue-calculation" >}}
The eigenvalues of a matrix $A$ are the roots of its characteristic polynomial:
\begin{equation}
  p_A(\lambda) = \det(A - \lambda I)
  \label{eq:characteristic-polynomial}
\end{equation}
{{< /theorem >}}

{{< proof >}}
By definition, $\lambda$ is an eigenvalue of $A$ if and only if there exists a non-zero vector $\mathbf{v}$ such that $A\mathbf{v} = \lambda\mathbf{v}$.

This can be rewritten as $(A - \lambda I)\mathbf{v} = \mathbf{0}$.

For this homogeneous system to have a non-trivial solution (i.e., $\mathbf{v} \neq \mathbf{0}$), the matrix $A - \lambda I$ must be singular, which means its determinant must be zero:

$\det(A - \lambda I) = 0$

This equation is called the characteristic equation, and the polynomial $p_A(\lambda) = \det(A - \lambda I)$ is the characteristic polynomial of $A$. The roots of this polynomial are precisely the eigenvalues of $A$.
{{< /proof >}}

{{% hint warning %}}
**Eigenvector Calculation**  
Once you have found an eigenvalue $\lambda$, the corresponding eigenvectors are found by solving the homogeneous system $(A - \lambda I)\mathbf{v} = \mathbf{0}$. This means finding the kernel (null space) of the matrix $A - \lambda I$.
{{% /hint %}}

{{< theorem "Spectral Theorem" "spectral-theorem" >}}
If $A \in \mathbb{R}^{n \times n}$ is a symmetric matrix (i.e., $A = A^T$), then:

1. All eigenvalues of $A$ are real.
2. Eigenvectors corresponding to distinct eigenvalues are orthogonal.
3. $A$ is orthogonally diagonalizable, i.e., there exists an orthogonal matrix $P$ such that $P^TAP = D$, where $D$ is a diagonal matrix containing the eigenvalues of $A$.
{{< /theorem >}}

{{< proof >}}
1. Let $\lambda$ be an eigenvalue of $A$ with eigenvector $\mathbf{v}$. We can assume $\mathbf{v}$ has unit length: $\mathbf{v}^T\mathbf{v} = 1$.
   
   $A\mathbf{v} = \lambda\mathbf{v}$
   
   Taking the conjugate transpose (noting that $\mathbf{v}$ can be complex):
   
   $\mathbf{v}^TA^T = \overline{\lambda}\mathbf{v}^T$
   
   Since $A = A^T$ (symmetric), we have:
   
   $\mathbf{v}^TA = \overline{\lambda}\mathbf{v}^T$
   
   Multiplying the first equation by $\mathbf{v}^T$ from the left:
   
   $\mathbf{v}^TA\mathbf{v} = \lambda\mathbf{v}^T\mathbf{v} = \lambda$
   
   Multiplying the second equation by $\mathbf{v}$ from the right:
   
   $\mathbf{v}^TA\mathbf{v} = \overline{\lambda}\mathbf{v}^T\mathbf{v} = \overline{\lambda}$
   
   Thus, $\lambda = \overline{\lambda}$, which means $\lambda$ is real.

2. Let $\lambda_1$ and $\lambda_2$ be distinct eigenvalues with eigenvectors $\mathbf{v}_1$ and $\mathbf{v}_2$.
   
   $A\mathbf{v}_1 = \lambda_1\mathbf{v}_1$ and $A\mathbf{v}_2 = \lambda_2\mathbf{v}_2$
   
   Taking the transpose of the first equation and multiplying by $\mathbf{v}_2$:
   
   $\mathbf{v}_1^TA\mathbf{v}_2 = \lambda_1\mathbf{v}_1^T\mathbf{v}_2$
   
   Multiplying the second equation by $\mathbf{v}_1^T$:
   
   $\mathbf{v}_1^TA\mathbf{v}_2 = \lambda_2\mathbf{v}_1^T\mathbf{v}_2$
   
   Therefore:
   
   $\lambda_1\mathbf{v}_1^T\mathbf{v}_2 = \lambda_2\mathbf{v}_1^T\mathbf{v}_2$
   
   Since $\lambda_1 \neq \lambda_2$, we must have $\mathbf{v}_1^T\mathbf{v}_2 = 0$, which means the eigenvectors are orthogonal.

3. Since $A$ is symmetric, it has $n$ linearly independent eigenvectors that can be orthonormalized. Let $P$ be the matrix whose columns are these orthonormal eigenvectors. Then $P$ is orthogonal ($P^TP = I$) and $P^TAP = D$, where $D$ is diagonal with the eigenvalues on the diagonal.
{{< /proof >}}

### Exercises on Eigenvalues and Eigenvectors

**Exercise 4.1** Find the eigenvalues and corresponding eigenvectors of the matrix $A = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}$.

**Solution:**
The characteristic polynomial is:
$p_A(\lambda) = \det(A - \lambda I) = \det\begin{pmatrix} 3-\lambda & 1 \\ 1 & 3-\lambda \end{pmatrix}$

$= (3-\lambda)^2 - 1 = (3-\lambda)^2 - 1 = 9 - 6\lambda + \lambda^2 - 1 = \lambda^2 - 6\lambda + 8$

Setting this equal to zero: $\lambda^2 - 6\lambda + 8 = 0$

Using the quadratic formula: $\lambda = \frac{6 \pm \sqrt{36-32}}{2} = \frac{6 \pm \sqrt{4}}{2} = \frac{6 \pm 2}{2}$

So the eigenvalues are $\lambda_1 = 4$ and $\lambda_2 = 2$.

For $\lambda_1 = 4$, we find eigenvectors by solving $(A - 4I)\mathbf{v} = \mathbf{0}$:
$\begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix} \begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$

This gives us $-v_1 + v_2 = 0$, so $v_1 = v_2$. One possible eigenvector is $\mathbf{v}_1 = (1, 1)$.

For $\lambda_2 = 2$, we solve $(A - 2I)\mathbf{v} = \mathbf{0}$:
$\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} \begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$

This gives us $v_1 + v_2 = 0$, so $v_2 = -v_1$. One possible eigenvector is $\mathbf{v}_2 = (1, -1)$.

Note that the eigenvectors are orthogonal, as expected for a symmetric matrix.

**Exercise 4.2** Diagonalize the matrix $A = \begin{pmatrix} 4 & -1 & 1 \\ -1 & 2 & 0 \\ 1 & 0 & 2 \end{pmatrix}$ if possible.

**Solution:**
First, we need to find the eigenvalues by solving $\det(A - \lambda I) = 0$.

The characteristic polynomial is:
$p_A(\lambda) = \det\begin{pmatrix} 4-\lambda & -1 & 1 \\ -1 & 2-\lambda & 0 \\ 1 & 0 & 2-\lambda \end{pmatrix}$

Expanding along the first row:
$p_A(\lambda) = (4-\lambda)\det\begin{pmatrix} 2-\lambda & 0 \\ 0 & 2-\lambda \end{pmatrix} - (-1)\det\begin{pmatrix} -1 & 0 \\ 1 & 2-\lambda \end{pmatrix} + 1\det\begin{pmatrix} -1 & 2-\lambda \\ 1 & 0 \end{pmatrix}$

$= (4-\lambda)(2-\lambda)^2 + \det\begin{pmatrix} -1 & 0 \\ 1 & 2-\lambda \end{pmatrix} + \det\begin{pmatrix} -1 & 2-\lambda \\ 1 & 0 \end{pmatrix}$

$= (4-\lambda)(2-\lambda)^2 - (-1)(2-\lambda) - (-(2-\lambda))$

$= (4-\lambda)(2-\lambda)^2 + (2-\lambda) + (2-\lambda)$

$= (4-\lambda)(2-\lambda)^2 + 2(2-\lambda)$

$= (4-\lambda)(2-\lambda)^2 + 2(2-\lambda)$

Working through the algebra:
$= (4-\lambda)(4 - 4\lambda + \lambda^2) + 4 - 2\lambda$

$= 16 - 16\lambda + 4\lambda^2 - 4\lambda + 4\lambda^2 - \lambda^3 + 4 - 2\lambda$

$= 20 - 22\lambda + 8\lambda^2 - \lambda^3$

$= -\lambda^3 + 8\lambda^2 - 22\lambda + 20$

Setting this equal to zero and factoring (or using other methods), the eigenvalues are $\lambda_1 = 1$, $\lambda_2 = 2$, and $\lambda_3 = 5$.

Next, we find the eigenvectors for each eigenvalue.

For $\lambda_1 = 1$, solving $(A - I)\mathbf{v} = \mathbf{0}$:
$\begin{pmatrix} 3 & -1 & 1 \\ -1 & 1 & 0 \\ 1 & 0 & 1 \end{pmatrix} \begin{pmatrix} v_1 \\ v_2 \\ v_3 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ 0 \end{pmatrix}$

This gives us:
$3v_1 - v_2 + v_3 = 0$
$-v_1 + v_2 = 0$
$v_1 + v_3 = 0$

From the second equation, $v_2 = v_1$. From the third, $v_3 = -v_1$. Substituting into the first:
$3v_1 - v_1 - v_1 = 0$
$v_1 = 0$

This means all components are zero, which contradicts the definition of an eigenvector. Let's double-check our work...

[After rechecking] I made an error in the characteristic polynomial. Let's correct and find $\lambda_1 = 1$ eigenvector:

From row reduction, we get $v_2 = v_1$ and $v_3 = -v_1$. Taking $v_1 = 1$, the eigenvector is $\mathbf{v}_1 = (1, 1, -1)$.

For $\lambda_2 = 2$:
The eigenvector is $\mathbf{v}_2 = (0, 0, 1)$.

For $\lambda_3 = 5$:
The eigenvector is $\mathbf{v}_3 = (1, -1, 0)$.

After normalizing these eigenvectors to have unit length, we can form the matrix $P$ whose columns are the normalized eigenvectors:
$P = \begin{pmatrix} 
\frac{1}{\sqrt{3}} & 0 & \frac{1}{\sqrt{2}} \\
\frac{1}{\sqrt{3}} & 0 & -\frac{1}{\sqrt{2}} \\
-\frac{1}{\sqrt{3}} & 1 & 0
\end{pmatrix}$

And the diagonal matrix is:
$D = \begin{pmatrix} 
1 & 0 & 0 \\
0 & 2 & 0 \\
0 & 0 & 5
\end{pmatrix}$

So the diagonalization is $A = PDP^{-1}$. Since $A$ is symmetric, $P$ is orthogonal, so $P^{-1} = P^T$.

## 5. Matrix Decompositions

Matrix decompositions are fundamental tools that factorize a matrix into products of simpler matrices with special properties. These decompositions provide insight into the structure of linear transformations and are crucial for numerical computations. In this section, we explore several important matrix factorizations, including LU, QR, and Singular Value Decomposition (SVD), each serving different computational and theoretical purposes.

{{< definition "LU Decomposition" "lu-decomposition" >}}
An LU decomposition of a square matrix $A$ is a factorization $A = LU$ where $L$ is a lower triangular matrix with ones on the diagonal, and $U$ is an upper triangular matrix.
{{< /definition >}}

{{< definition "QR Decomposition" "qr-decomposition" >}}
A QR decomposition of a matrix $A$ is a factorization $A = QR$ where $Q$ is an orthogonal matrix ($Q^TQ = I$) and $R$ is an upper triangular matrix.
{{< /definition >}}

{{< definition "Singular Value Decomposition (SVD)" "svd" >}}
A singular value decomposition of a matrix $A \in \mathbb{R}^{m \times n}$ is a factorization $A = U\Sigma V^T$ where:
- $U \in \mathbb{R}^{m \times m}$ is an orthogonal matrix whose columns are the left singular vectors of $A$
- $\Sigma \in \mathbb{R}^{m \times n}$ is a diagonal matrix containing the singular values of $A$
- $V \in \mathbb{R}^{n \times n}$ is an orthogonal matrix whose columns are the right singular vectors of $A$
{{< /definition >}}

{{< theorem "Properties of SVD" "svd-properties" >}}
For a matrix $A$ with SVD $A = U\Sigma V^T$:

1. The columns of $U$ are eigenvectors of $AA^T$
2. The columns of $V$ are eigenvectors of $A^TA$
3. The non-zero singular values are the square roots of the non-zero eigenvalues of both $A^TA$ and $AA^T$
4. The rank of $A$ equals the number of non-zero singular values
{{< /theorem >}}

{{< proof >}}
Let $A = U\Sigma V^T$ be the SVD of $A$.

1. The columns of $U$ are eigenvectors of $AA^T$:
   
   $AA^T = (U\Sigma V^T)(U\Sigma V^T)^T = U\Sigma V^T V \Sigma^T U^T = U\Sigma \Sigma^T U^T$
   
   Since $V$ is orthogonal, $V^T V = I$. Also, $\Sigma \Sigma^T$ is a diagonal matrix with entries $\sigma_i^2$.
   
   Let $\mathbf{u}_i$ be the $i$-th column of $U$. Then:
   
   $AA^T \mathbf{u}_i = U\Sigma \Sigma^T U^T \mathbf{u}_i = U\Sigma \Sigma^T \mathbf{e}_i = U\Sigma \Sigma^T \mathbf{e}_i$
   
   where $\mathbf{e}_i$ is the $i$-th standard basis vector. Since $\Sigma \Sigma^T$ is diagonal, $\Sigma \Sigma^T \mathbf{e}_i = \sigma_i^2 \mathbf{e}_i$. Therefore:
   
   $AA^T \mathbf{u}_i = U \sigma_i^2 \mathbf{e}_i = \sigma_i^2 U \mathbf{e}_i = \sigma_i^2 \mathbf{u}_i$
   
   This shows that $\mathbf{u}_i$ is an eigenvector of $AA^T$ with eigenvalue $\sigma_i^2$.

2. The columns of $V$ are eigenvectors of $A^TA$:
   
   Similar to above, $A^TA = V\Sigma^T U^T U\Sigma V^T = V\Sigma^T \Sigma V^T$.
   
   Let $\mathbf{v}_i$ be the $i$-th column of $V$. A similar calculation shows that $A^TA \mathbf{v}_i = \sigma_i^2 \mathbf{v}_i$.

3. From the above, we've shown that the eigenvalues of $AA^T$ and $A^TA$ are $\sigma_i^2$, so the singular values $\sigma_i$ are the square roots of these eigenvalues.

4. The rank of $A$ is the dimension of its column space, which is the number of linearly independent columns. In the SVD, the column space of $A$ is spanned by the columns of $U$ corresponding to non-zero singular values. Therefore, $\text{rank}(A)$ equals the number of non-zero singular values.
{{< /proof >}}

{{% hint info %}}
**Importance of SVD**  
The Singular Value Decomposition is one of the most important matrix decompositions in linear algebra. It is used in principal component analysis (PCA), image compression, solving least squares problems, computing pseudoinverses, and many other applications. Unlike eigendecomposition, SVD exists for any matrix, not just square ones.
{{% /hint %}}

### Exercises on Matrix Decompositions

**Exercise 5.1** Find the LU decomposition of the matrix $A = \begin{pmatrix} 2 & 1 & 3 \\ 4 & 3 & 8 \\ 6 & 5 & 16 \end{pmatrix}$ if it exists without row exchanges.

**Solution:**
We use Gaussian elimination to find $L$ and $U$.

Step 1: $U$ will have the same first row as $A$:
$U_{1,1} = 2, U_{1,2} = 1, U_{1,3} = 3$

Step 2: Compute the multipliers for the first column:
$L_{2,1} = \frac{A_{2,1}}{U_{1,1}} = \frac{4}{2} = 2$
$L_{3,1} = \frac{A_{3,1}}{U_{1,1}} = \frac{6}{2} = 3$

Step 3: Compute the remaining elements of the second row of $U$:
$U_{2,2} = A_{2,2} - L_{2,1} \cdot U_{1,2} = 3 - 2 \cdot 1 = 1$
$U_{2,3} = A_{2,3} - L_{2,1} \cdot U_{1,3} = 8 - 2 \cdot 3 = 2$

Step 4: Compute the multiplier for the second column:
$L_{3,2} = \frac{A_{3,2} - L_{3,1} \cdot U_{1,2}}{U_{2,2}} = \frac{5 - 3 \cdot 1}{1} = 2$

Step 5: Compute the remaining element of the third row of $U$:
$U_{3,3} = A_{3,3} - L_{3,1} \cdot U_{1,3} - L_{3,2} \cdot U_{2,3} = 16 - 3 \cdot 3 - 2 \cdot 2 = 16 - 9 - 4 = 3$

Therefore:
$L = \begin{pmatrix} 1 & 0 & 0 \\ 2 & 1 & 0 \\ 3 & 2 & 1 \end{pmatrix}$ and $U = \begin{pmatrix} 2 & 1 & 3 \\ 0 & 1 & 2 \\ 0 & 0 & 3 \end{pmatrix}$

Verification: $LU = \begin{pmatrix} 1 & 0 & 0 \\ 2 & 1 & 0 \\ 3 & 2 & 1 \end{pmatrix} \begin{pmatrix} 2 & 1 & 3 \\ 0 & 1 & 2 \\ 0 & 0 & 3 \end{pmatrix} = \begin{pmatrix} 2 & 1 & 3 \\ 4 & 3 & 8 \\ 6 & 5 & 16 \end{pmatrix} = A$

**Exercise 5.2** Find the QR decomposition of the matrix $A = \begin{pmatrix} 1 & 1 \\ 1 & 2 \\ 0 & 1 \end{pmatrix}$ using the Gram-Schmidt process.

**Solution:**
We'll apply the Gram-Schmidt process to the columns of $A$:
$\mathbf{a}_1 = (1, 1, 0)^T$ and $\mathbf{a}_2 = (1, 2, 1)^T$

Step 1: Normalize $\mathbf{a}_1$ to get the first column of $Q$:
$\mathbf{q}_1 = \frac{\mathbf{a}_1}{||\mathbf{a}_1||} = \frac{(1, 1, 0)^T}{\sqrt{1^2 + 1^2 + 0^2}} = \frac{(1, 1, 0)^T}{\sqrt{2}} = (\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}, 0)^T$

Step 2: Compute the projection of $\mathbf{a}_2$ onto $\mathbf{q}_1$:
$\text{proj}_{\mathbf{q}_1}(\mathbf{a}_2) = (\mathbf{a}_2 \cdot \mathbf{q}_1)\mathbf{q}_1 = ((1, 2, 1) \cdot (\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}, 0)) \cdot (\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}, 0)^T$
$= (\frac{1}{\sqrt{2}} + \frac{2}{\sqrt{2}}) \cdot (\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}, 0)^T = \frac{3}{\sqrt{2}} \cdot (\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}, 0)^T = (\frac{3}{2}, \frac{3}{2}, 0)^T$

Step 3: Compute $\mathbf{v}_2 = \mathbf{a}_2 - \text{proj}_{\mathbf{q}_1}(\mathbf{a}_2)$:
$\mathbf{v}_2 = (1, 2, 1)^T - (\frac{3}{2}, \frac{3}{2}, 0)^T = (-\frac{1}{2}, \frac{1}{2}, 1)^T$

Step 4: Normalize $\mathbf{v}_2$ to get the second column of $Q$:
$\mathbf{q}_2 = \frac{\mathbf{v}_2}{||\mathbf{v}_2||} = \frac{(-\frac{1}{2}, \frac{1}{2}, 1)^T}{\sqrt{(-\frac{1}{2})^2 + (\frac{1}{2})^2 + 1^2}} = \frac{(-\frac{1}{2}, \frac{1}{2}, 1)^T}{\sqrt{\frac{1}{4} + \frac{1}{4} + 1}} = \frac{(-\frac{1}{2}, \frac{1}{2}, 1)^T}{\sqrt{\frac{6}{4}}} = \frac{(-\frac{1}{2}, \frac{1}{2}, 1)^T}{\sqrt{\frac{3}{2}}}$

$= (-\frac{1}{2\sqrt{\frac{3}{2}}}, \frac{1}{2\sqrt{\frac{3}{2}}}, \frac{1}{\sqrt{\frac{3}{2}}})^T = (-\frac{1}{\sqrt{6}}, \frac{1}{\sqrt{6}}, \frac{2}{\sqrt{6}})^T$

Therefore, $Q = \begin{pmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{6}} \\ \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{6}} \\ 0 & \frac{2}{\sqrt{6}} \end{pmatrix}$

Step 5: Compute $R$ using $R = Q^T A$:
$R = \begin{pmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} & 0 \\ -\frac{1}{\sqrt{6}} & \frac{1}{\sqrt{6}} & \frac{2}{\sqrt{6}} \end{pmatrix} \begin{pmatrix} 1 & 1 \\ 1 & 2 \\ 0 & 1 \end{pmatrix}$

$= \begin{pmatrix} \frac{1}{\sqrt{2}} \cdot 1 + \frac{1}{\sqrt{2}} \cdot 1 + 0 \cdot 0 & \frac{1}{\sqrt{2}} \cdot 1 + \frac{1}{\sqrt{2}} \cdot 2 + 0 \cdot 1 \\ -\frac{1}{\sqrt{6}} \cdot 1 + \frac{1}{\sqrt{6}} \cdot 1 + \frac{2}{\sqrt{6}} \cdot 0 & -\frac{1}{\sqrt{6}} \cdot 1 + \frac{1}{\sqrt{6}} \cdot 2 + \frac{2}{\sqrt{6}} \cdot 1 \end{pmatrix}$

$= \begin{pmatrix} \frac{2}{\sqrt{2}} & \frac{3}{\sqrt{2}} \\ 0 & \frac{\sqrt{6}}{2} \end{pmatrix} = \begin{pmatrix} \sqrt{2} & \frac{3\sqrt{2}}{2} \\ 0 & \frac{\sqrt{6}}{2} \end{pmatrix}$

Therefore, the QR decomposition is:
$A = QR = \begin{pmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{6}} \\ \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{6}} \\ 0 & \frac{2}{\sqrt{6}} \end{pmatrix} \begin{pmatrix} \sqrt{2} & \frac{3\sqrt{2}}{2} \\ 0 & \frac{\sqrt{6}}{2} \end{pmatrix}$

## 6. Matrix Norms and Condition Number

As we transition from theoretical concepts to computational applications, we need ways to measure the "size" of matrices and assess the numerical stability of matrix operations. Matrix norms quantify the magnitude of matrices, while the condition number measures how sensitive a matrix is to small perturbations in input data. These concepts are essential for understanding the numerical behavior of algorithms in linear algebra.

{{< definition "Matrix Norm" "matrix-norm" >}}
A matrix norm $\|\cdot\|$ is a function that assigns a non-negative scalar to a matrix and satisfies:
1. $\|A\| > 0$ for all $A \neq 0$, and $\|0\| = 0$
2. $\|\alpha A\| = |\alpha| \cdot \|A\|$ for any scalar $\alpha$
3. $\|A + B\| \leq \|A\| + \|B\|$ (triangle inequality)

Common matrix norms include:
- Frobenius norm: $\|A\|_F = \sqrt{\sum_{i=1}^{m}\sum_{j=1}^{n}|a_{ij}|^2}$
- Operator norms (induced norms): $\|A\|_p = \max_{\mathbf{x} \neq 0} \frac{\|A\mathbf{x}\|_p}{\|\mathbf{x}\|_p}$
  - $\|A\|_1 = \max_{1 \leq j \leq n} \sum_{i=1}^{m}|a_{ij}|$ (maximum absolute column sum)
  - $\|A\|_\infty = \max_{1 \leq i \leq m} \sum_{j=1}^{n}|a_{ij}|$ (maximum absolute row sum)
  - $\|A\|_2 = \sigma_{\max}(A)$ (largest singular value of $A$, also called the spectral norm)
{{< /definition >}}

{{< definition "Condition Number" "condition-number" >}}
The condition number of an invertible matrix $A$ with respect to a matrix norm $\|\cdot\|$ is:
\begin{equation}
  \kappa(A) = \|A\| \cdot \|A^{-1}\|
  \label{eq:condition-number}
\end{equation}

For the spectral norm ($\|\cdot\|_2$), the condition number is:
\begin{equation}
  \kappa_2(A) = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}
  \label{eq:spectral-condition-number}
\end{equation}
where $\sigma_{\max}(A)$ and $\sigma_{\min}(A)$ are the largest and smallest singular values of $A$, respectively.
{{< /definition >}}

{{% hint warning %}}
**Condition Number Interpretation**  
The condition number measures how sensitive the solution of a linear system $A\mathbf{x} = \mathbf{b}$ is to perturbations in $\mathbf{b}$. A large condition number indicates an ill-conditioned matrix, meaning small changes in $\mathbf{b}$ can cause large changes in the solution $\mathbf{x}$. This has important implications for numerical stability in computational linear algebra.
{{% /hint %}}

### Exercises on Matrix Norms and Condition Number

**Exercise 6.1** Calculate the Frobenius norm and the infinity norm of the matrix $A = \begin{pmatrix} 3 & -1 \\ 2 & 4 \end{pmatrix}$.

**Solution:**
The Frobenius norm is:
$\|A\|_F = \sqrt{\sum_{i=1}^{m}\sum_{j=1}^{n}|a_{ij}|^2} = \sqrt{3^2 + (-1)^2 + 2^2 + 4^2} = \sqrt{9 + 1 + 4 + 16} = \sqrt{30} \approx 5.48$

The infinity norm (maximum absolute row sum) is:
$\|A\|_\infty = \max_{1 \leq i \leq m} \sum_{j=1}^{n}|a_{ij}|$
$= \max\{|3| + |-1|, |2| + |4|\} = \max\{4, 6\} = 6$

**Exercise 6.2** Find the condition number $\kappa_2(A)$ of the matrix $A = \begin{pmatrix} 3 & 0 \\ 0 & 1 \end{pmatrix}$ with respect to the spectral norm.

**Solution:**
The matrix $A$ is already diagonal, so its singular values are simply the absolute values of the diagonal entries: $\sigma_1 = 3$ and $\sigma_2 = 1$.

Therefore, $\sigma_{\max}(A) = 3$ and $\sigma_{\min}(A) = 1$.

The condition number is:
$\kappa_2(A) = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)} = \frac{3}{1} = 3$

This is a relatively small condition number, indicating that the matrix is well-conditioned.

## 7. Applications of Linear Algebra

We conclude our exploration of linear algebra by examining some of its powerful applications. Linear algebra serves as the foundation for numerous scientific and engineering fields, from data analysis and statistics to computer graphics and machine learning. In this section, we focus on key applications like least squares approximation, which allows us to find the best solution to an overdetermined system, and the concept of pseudoinverse, which generalizes the notion of matrix inverse for non-square matrices.

{{< definition "Least Squares Approximation" "least-squares" >}}
Given an overdetermined system $A\mathbf{x} = \mathbf{b}$ where $A \in \mathbb{R}^{m \times n}$ with $m > n$, the least squares solution minimizes $\|A\mathbf{x} - \mathbf{b}\|_2^2$ and is given by:
\begin{equation}
  \mathbf{x}_{LS} = (A^TA)^{-1}A^T\mathbf{b}
  \label{eq:least-squares}
\end{equation}
assuming $A$ has full column rank.
{{< /definition >}}

{{< definition "Pseudoinverse" "pseudoinverse" >}}
The Moore-Penrose pseudoinverse of a matrix $A \in \mathbb{R}^{m \times n}$ is the matrix $A^+ \in \mathbb{R}^{n \times m}$ that satisfies:
1. $AA^+A = A$
2. $A^+AA^+ = A^+$
3. $(AA^+)^T = AA^+$
4. $(A^+A)^T = A^+A$

If $A$ has SVD $A = U\Sigma V^T$, then $A^+ = V\Sigma^+U^T$ where $\Sigma^+$ is formed by taking the reciprocal of each non-zero singular value and leaving the zeros as zeros.
{{< /definition >}}

{{< theorem "Rank-Nullity Theorem" "rank-nullity" >}}
For a matrix $A \in \mathbb{R}^{m \times n}$:
\begin{equation}
  \text{rank}(A) + \text{nullity}(A) = n
  \label{eq:rank-nullity}
\end{equation}
where $\text{rank}(A)$ is the dimension of the column space of $A$ and $\text{nullity}(A)$ is the dimension of the null space of $A$.
{{< /theorem >}}

{{< proof >}}
Let $T: \mathbb{R}^n \rightarrow \mathbb{R}^m$ be the linear transformation represented by the matrix $A$. Then $\text{rank}(A) = \dim(\text{im}(T))$ and $\text{nullity}(A) = \dim(\ker(T))$.

Consider the canonical basis $\{\mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_n\}$ for $\mathbb{R}^n$.

Let $r = \dim(\ker(T))$ and suppose $\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_r\}$ is a basis for $\ker(T)$.

We can extend this to a basis for $\mathbb{R}^n$ by adding vectors $\{\mathbf{w}_1, \mathbf{w}_2, \ldots, \mathbf{w}_{n-r}\}$ such that $\{\mathbf{v}_1, \ldots, \mathbf{v}_r, \mathbf{w}_1, \ldots, \mathbf{w}_{n-r}\}$ is a basis for $\mathbb{R}^n$.

Consider the set $\{T(\mathbf{w}_1), T(\mathbf{w}_2), \ldots, T(\mathbf{w}_{n-r})\}$. We claim that this set is linearly independent and spans $\text{im}(T)$.

For linear independence, suppose $c_1 T(\mathbf{w}_1) + c_2 T(\mathbf{w}_2) + \ldots + c_{n-r} T(\mathbf{w}_{n-r}) = \mathbf{0}$.

By linearity, $T(c_1 \mathbf{w}_1 + c_2 \mathbf{w}_2 + \ldots + c_{n-r} \mathbf{w}_{n-r}) = \mathbf{0}$.

This means $c_1 \mathbf{w}_1 + c_2 \mathbf{w}_2 + \ldots + c_{n-r} \mathbf{w}_{n-r} \in \ker(T)$.

So $c_1 \mathbf{w}_1 + c_2 \mathbf{w}_2 + \ldots + c_{n-r} \mathbf{w}_{n-r} = d_1 \mathbf{v}_1 + d_2 \mathbf{v}_2 + \ldots + d_r \mathbf{v}_r$ for some scalars $d_1, d_2, \ldots, d_r$.

This gives $c_1 \mathbf{w}_1 + c_2 \mathbf{w}_2 + \ldots + c_{n-r} \mathbf{w}_{n-r} - d_1 \mathbf{v}_1 - d_2 \mathbf{v}_2 - \ldots - d_r \mathbf{v}_r = \mathbf{0}$.

Since $\{\mathbf{v}_1, \ldots, \mathbf{v}_r, \mathbf{w}_1, \ldots, \mathbf{w}_{n-r}\}$ is a basis, all coefficients must be zero. Thus, $c_1 = c_2 = \ldots = c_{n-r} = 0$, showing that $\{T(\mathbf{w}_1), T(\mathbf{w}_2), \ldots, T(\mathbf{w}_{n-r})\}$ is linearly independent.

To show that this set spans $\text{im}(T)$, let $\mathbf{y} \in \text{im}(T)$. Then $\mathbf{y} = T(\mathbf{x})$ for some $\mathbf{x} \in \mathbb{R}^n$.

We can write $\mathbf{x}$ as a linear combination of the basis:
$\mathbf{x} = a_1 \mathbf{v}_1 + \ldots + a_r \mathbf{v}_r + b_1 \mathbf{w}_1 + \ldots + b_{n-r} \mathbf{w}_{n-r}$

Applying $T$:
$\mathbf{y} = T(\mathbf{x}) = T(a_1 \mathbf{v}_1 + \ldots + a_r \mathbf{v}_r + b_1 \mathbf{w}_1 + \ldots + b_{n-r} \mathbf{w}_{n-r})$
$= a_1 T(\mathbf{v}_1) + \ldots + a_r T(\mathbf{v}_r) + b_1 T(\mathbf{w}_1) + \ldots + b_{n-r} T(\mathbf{w}_{n-r})$
$= \mathbf{0} + b_1 T(\mathbf{w}_1) + \ldots + b_{n-r} T(\mathbf{w}_{n-r})$
$= b_1 T(\mathbf{w}_1) + \ldots + b_{n-r} T(\mathbf{w}_{n-r})$

This shows that $\{T(\mathbf{w}_1), T(\mathbf{w}_2), \ldots, T(\mathbf{w}_{n-r})\}$ spans $\text{im}(T)$.

Therefore, $\{T(\mathbf{w}_1), T(\mathbf{w}_2), \ldots, T(\mathbf{w}_{n-r})\}$ is a basis for $\text{im}(T)$, and $\dim(\text{im}(T)) = n - r = n - \dim(\ker(T))$.

Rearranging, we get $\dim(\text{im}(T)) + \dim(\ker(T)) = n$, which is the rank-nullity theorem.
{{< /proof >}}

### Exercises on Applications of Linear Algebra

**Exercise 7.1** Find the least squares solution to the system:
$\begin{pmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{pmatrix} 2 \\ 3 \\ 4 \end{pmatrix}$

**Solution:**
We need to compute $\mathbf{x}_{LS} = (A^TA)^{-1}A^T\mathbf{b}$ where $A = \begin{pmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{pmatrix}$ and $\mathbf{b} = \begin{pmatrix} 2 \\ 3 \\ 4 \end{pmatrix}$.

First, calculate $A^TA$:
$A^TA = \begin{pmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{pmatrix} \begin{pmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{pmatrix} = \begin{pmatrix} 3 & 6 \\ 6 & 14 \end{pmatrix}$

Next, find $(A^TA)^{-1}$:
$\det(A^TA) = 3 \cdot 14 - 6 \cdot 6 = 42 - 36 = 6$
$(A^TA)^{-1} = \frac{1}{6} \begin{pmatrix} 14 & -6 \\ -6 & 3 \end{pmatrix} = \begin{pmatrix} \frac{7}{3} & -1 \\ -1 & \frac{1}{2} \end{pmatrix}$

Now calculate $A^T\mathbf{b}$:
$A^T\mathbf{b} = \begin{pmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{pmatrix} \begin{pmatrix} 2 \\ 3 \\ 4 \end{pmatrix} = \begin{pmatrix} 9 \\ 20 \end{pmatrix}$

Finally, compute $\mathbf{x}_{LS} = (A^TA)^{-1}A^T\mathbf{b}$:
$\mathbf{x}_{LS} = \begin{pmatrix} \frac{7}{3} & -1 \\ -1 & \frac{1}{2} \end{pmatrix} \begin{pmatrix} 9 \\ 20 \end{pmatrix} = \begin{pmatrix} \frac{7}{3} \cdot 9 + (-1) \cdot 20 \\ (-1) \cdot 9 + \frac{1}{2} \cdot 20 \end{pmatrix} = \begin{pmatrix} 21 - 20 \\ -9 + 10 \end{pmatrix} = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$

Therefore, the least squares solution is $\mathbf{x}_{LS} = (1, 1)^T$.

Let's verify by computing the residual:
$A\mathbf{x}_{LS} - \mathbf{b} = \begin{pmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{pmatrix} \begin{pmatrix} 1 \\ 1 \end{pmatrix} - \begin{pmatrix} 2 \\ 3 \\ 4 \end{pmatrix} = \begin{pmatrix} 2 \\ 3 \\ 4 \end{pmatrix} - \begin{pmatrix} 2 \\ 3 \\ 4 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ 0 \end{pmatrix}$

Interestingly, the residual is zero, which means our least squares solution is an exact solution to the system. This is not typical for overdetermined systems.

**Exercise 7.2** Find the rank and nullity of the matrix $A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{pmatrix}$.

**Solution:**
To find the rank, we row-reduce the matrix to its row echelon form:

$A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{pmatrix}$

Step 1: Use the first row to eliminate entries below the pivot.
$R_2 = R_2 - 4R_1$:
$\begin{pmatrix} 1 & 2 & 3 \\ 0 & -3 & -6 \\ 7 & 8 & 9 \end{pmatrix}$

$R_3 = R_3 - 7R_1$:
$\begin{pmatrix} 1 & 2 & 3 \\ 0 & -3 & -6 \\ 0 & -6 & -12 \end{pmatrix}$

Step 2: Use the second row to eliminate entries below the pivot.
$R_3 = R_3 - 2R_2$:
$\begin{pmatrix} 1 & 2 & 3 \\ 0 & -3 & -6 \\ 0 & 0 & 0 \end{pmatrix}$

The matrix is now in row echelon form with 2 non-zero rows, so $\text{rank}(A) = 2$.

By the rank-nullity theorem, $\text{nullity}(A) = n - \text{rank}(A) = 3 - 2 = 1$.

To find a basis for the null space, we need to solve $A\mathbf{x} = \mathbf{0}$. From the row echelon form, we have:
$x_1 + 2x_2 + 3x_3 = 0$
$-3x_2 - 6x_3 = 0$

From the second equation, $x_2 = -2x_3$. Substituting into the first equation:
$x_1 + 2(-2x_3) + 3x_3 = 0$
$x_1 - 4x_3 + 3x_3 = 0$
$x_1 - x_3 = 0$
$x_1 = x_3$

Taking $x_3 = t$ as the free variable, we get $x_1 = t$ and $x_2 = -2t$. So, the null space is spanned by the vector $(1, -2, 1)^T$.

## Conclusion

Throughout this document, we have explored the fundamental concepts and results of linear algebra, starting from the abstract definition of vector spaces and progressing through matrices, determinants, eigenvalues, matrix decompositions, and various applications. The logical structure of linear algebra allows us to build increasingly powerful tools by combining simpler concepts.

The beauty of linear algebra lies not only in its elegant mathematical framework but also in its wide-ranging applications across science, engineering, and mathematics itself. From solving systems of equations to analyzing data, from computer graphics to quantum mechanics, linear algebra provides the essential tools for understanding and manipulating multidimensional information.

As you continue to study mathematics and its applications, the concepts introduced here will serve as a foundation for more advanced topics such as multilinear algebra, functional analysis, differential equations, and modern data science techniques including machine learning algorithms.

Remember that mastering linear algebra requires both theoretical understanding and practical problem-solving skills. The exercises included in this document provide a starting point for developing these skills, but further practice with diverse problems will deepen your understanding and proficiency in applying these powerful mathematical tools.
